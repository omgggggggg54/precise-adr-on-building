import os
import datetime
import argparse
import csv
import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

from args import *
from dataset.please import DataModule
from models.hetero_cl import *
from root_utils import seed_everything, build_save_path

# main_eval.py 或 main.py 最开头（import 后）
import torch
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
torch.serialization.add_safe_globals([argparse.Namespace])


def setup_warning_filters():
    """全局关闭 warnings，保证终端输出干净。"""
    warnings.filterwarnings("ignore")


setup_warning_filters()

def build_dataset_func(args):
    """根据参数构建数据模块，并把图结构相关维度回填到 `args`。"""
    if args.dataset in ["gender", "age", "all"]:
        # 这里直接把数据集类型同步到 se_type，后续 DataModule 会按这个字段加载对应文件。
        args.se_type = args.dataset

        # n_layer 控制邻居采样层数，通常与模型的 GNN 层数保持一致。
        # batch_size 表示每个 batch 选取多少个 patient 种子节点。
        # split 决定 train/val/test 的划分方式。
        # n_data 为读取的病例数量，0 表示读取全部。
        # add_SE 控制是否把训练标签额外构造成 patient -> SE 边。
        datamodule = DataModule(
            n_layer=args.n_gnn,
            batch_size=args.batch_size,
            split=args.split,
            n_data=args.n_data,
            add_SE=args.add_SE,
            # 这里显式使用 processed 缓存，避免默认值变化影响复现结果。
            use_processed=False,
            args=args
        )
    else:
        raise RuntimeError(f"Don't support Dataset:{args.dataset}")

    # setup 会真正构建图、切分掩码，并准备 dataloader 所需的数据对象。
    datamodule.setup()
    data = datamodule.data#图

    # 这几个起止位置对应 patient BOW 特征里 indication / drug 特征片段的索引范围。
    # 后续如果需要做特征解释或裁剪，会依赖这些边界信息。
    args.i_start = datamodule.i_start
    args.i_end = datamodule.i_end
    args.d_start = datamodule.d_start
    args.d_end = datamodule.d_end

    # metadata 是 PyG HeteroData 的元信息，包含节点类型和边类型。
    args.metadata = data.metadata()

    # 记录每种节点类型输入特征的维度，供异构模型初始化时使用。
    args.in_dim_dict = {n_t: data[n_t].x.size(0) for n_t in data.metadata()[0]}

    # patient.y 的第二维就是多标签分类的类别数，也就是 SE 的数量。
    args.out_dim = data["patient"].y.size(1)

    # num_info 表示 patient 节点中“个人信息特征”的前缀长度。
    args.num_info = data["patient"].num_info
    args.num_node = data.num_nodes

    # patient.bow_feat 的列数就是统一 BOW 特征空间的维度。
    args.num_feat = args.in_dim = data["patient"].bow_feat.size(1)
    # 药物结构特征维度单独记录，供模型侧做结构投影。
    if "struct_feat" in data["drug"]:
        args.drug_struct_dim = data["drug"].struct_feat.size(1)
    else:
        args.drug_struct_dim = 0

    # 打印划分结果，便于确认当前 split 是否符合预期。
    print("train data size:", datamodule.data["patient"].train_mask.sum())
    print("val data size:", datamodule.data["patient"].val_mask.sum())
    print("test data size:", datamodule.data["patient"].test_mask.sum())

    return datamodule


def main(args, other_callbacks=[], dataset_func=build_dataset_func, model_wrapper=ContrastiveWrapper,
         remove_file=False):
    """执行一次完整的训练、验证、测试与预测导出流程。"""
    print(args.seed)

    # 固定随机种子，尽量让数据划分、采样和训练结果可复现。
    seed_everything(args.seed)

    # 先给本次训练生成唯一标识，并创建分类输出目录。
    args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_root = os.path.join("outputs", args.dataset, args.model_name, f"seed_{args.seed}", args.run_id)
    ckpt_dir = os.path.join(save_root, "checkpoints")
    pred_dir = os.path.join(save_root, "predictions")
    report_dir = os.path.join(save_root, "reports")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 先构建数据模块，再根据已经补齐的 args 初始化模型。
    datamodule = dataset_func(args)
    model = model_wrapper(model_name=args.model_name, args=args)
    # 统一按 args.device 判定训练设备，避免 main_eval 漏传参数时走错设备。
    accelerator = "gpu" if ("cuda" in str(args.device).lower() and torch.cuda.is_available()) else "cpu"
    devices = 1
    # 这里打印一次关键设备信息，日志里能直接确认是否真的在用 GPU。
    print(f"trainer device => args.device={args.device}, accelerator={accelerator}, devices={devices}")

    # 监控 val_auc，保存验证集最优模型。
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        save_top_k=1,
        mode='max',
        dirpath=ckpt_dir,
        filename="best-{epoch:03d}-{val_auc:.4f}",
        auto_insert_metric_name=False
    )

    # 如果验证指标长期不提升，则提前结束训练。
    early_stop_callback = EarlyStopping(monitor='val_auc', mode="max", patience=args.patient)

    callbacks = [checkpoint_callback, early_stop_callback]
    # 强制开启细粒度进度条，便于在终端持续看到训练推进。
    callbacks.append(TQDMProgressBar(refresh_rate=1))
    callbacks.extend(other_callbacks)

    # Lightning Trainer 负责统一调度训练、验证、测试和预测流程。
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.max_epochs,
        # max_epochs=1,
        # auto_select_gpus=True,
        check_val_every_n_epoch=args.eval_step,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        # logger=False
    )

    # 正式训练。
    trainer.fit(model, datamodule)

    print(args.model_name)

    # 训练结束后，再单独跑一次验证集，拿到最终最佳模型对应的验证指标。
    # 显式指定 best checkpoint，避免 Lightning 走隐式回退逻辑。
    max_val_score = trainer.validate(
        model=model,
        dataloaders=datamodule.val_dataloader(),
        ckpt_path=checkpoint_callback.best_model_path
    )
    val_score = max_val_score[0]['val_auc']

    # 手动保存最后一个 epoch 的模型（与 best 模型分开存）。
    args.save_path = os.path.join(ckpt_dir, "last_" + build_save_path(args, val_score))
    trainer.save_checkpoint(args.save_path)
    args.best_ckpt_path = checkpoint_callback.best_model_path

    # 测试集评估。
    max_test_score = trainer.test(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=args.best_ckpt_path)
    test_score = max_test_score[0] if len(max_test_score) > 0 else {}

    if remove_file:
        # 某些场景只关心分数，不保留 ckpt 文件。
        os.remove(args.save_path)
    else:
        # 额外导出测试集预测结果，方便离线分析。
        res = trainer.predict(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=args.best_ckpt_path)
        pred_path = os.path.join(pred_dir, f"test_pred_{args.run_id}.pth")
        torch.save(res, pred_path)

    # 保存一份简洁 txt：包含关键结果和关键参数。
    report_path = os.path.join(report_dir, f"train_report_{args.run_id}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"run_id: {args.run_id}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"model_name: {args.model_name}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"best_ckpt: {args.best_ckpt_path}\n")
        f.write(f"last_ckpt: {args.save_path}\n")
        f.write(f"val_auc: {val_score}\n")
        f.write(f"test_metrics: {test_score}\n")
        f.write("\n")
        f.write("params:\n")
        for k in [
            "batch_size", "n_gnn", "n_mlp", "max_epochs", "eval_step", "split",
            "n_data", "num_neigh", "lr", "weight_decay", "dropout", "add_SE", "device",
            "use_drug_struct", "drug_encoder_type", "drug_struct_dim", "drug_smiles_csv",
            "molformer_feat_path", "use_time_feature", "time_dim"
        ]:
            f.write(f"{k}: {getattr(args, k, None)}\n")
    # 统一维护一个总索引表，后续找最优权重/指标时不用翻目录。
    run_index_path = os.path.join("outputs", "run_index.csv")
    os.makedirs(os.path.dirname(run_index_path), exist_ok=True)
    header = [
        "run_id", "dataset", "model_name", "seed", "val_auc", "test_auc",
        "best_ckpt", "last_ckpt", "report_path"
    ]
    row = [
        args.run_id, args.dataset, args.model_name, args.seed, val_score, test_score.get("test_auc"),
        args.best_ckpt_path, args.save_path, report_path
    ]
    need_header = not os.path.exists(run_index_path)
    with open(run_index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"报告已保存: {report_path}")
    print(f"索引已追加: {run_index_path}")

    return max_val_score, max_test_score


def main_predict(args, dataset_func=build_dataset_func, model_wrapper=BasicModelWrapper, ckpt_path=None):
    """加载已有 checkpoint，并对测试集做预测导出。"""
    seed_everything(args.seed)
    datamodule = dataset_func(args)

    # 这里直接从 checkpoint 反序列化 LightningModule。
    model = model_wrapper.load_from_checkpoint(checkpoint_path=ckpt_path)

    trainer = Trainer(
        accelerator='gpu' if "cuda" in args.device else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.eval_step,
        num_sanity_val_steps=0,
    )

    # 预测结果统一归档到 outputs/predict 下。
    predict_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = os.path.join("outputs", "predict", args.dataset, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    args.save_path = os.path.join(save_dir, f"predict_{predict_id}.pth")
    res = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    torch.save(res, args.save_path)

    return res


if __name__ == "__main__":
    # 解析命令行参数与 yaml 配置。
    parser = argparse.ArgumentParser(description="PreciseADR")
    register_args(parser)
    args = parse_args_and_yaml(parser)
    dataset = args.dataset

    # 如果用户指定了 cuda，但当前环境没有可用 GPU，就自动回退到 cpu。
    if "cuda" in args.device:
        args.device = args.device if torch.cuda.is_available() else "cpu"

    seed = args.seed
    res_dict = {}
    job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = os.path.join("outputs", dataset, "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        # 当前脚本默认只跑 1 次实验，但保留了多次重复实验的循环结构。
        for i in range(1):
            for model_name in ["PreciseADR_HGT"]:
                args.model_name = model_name
                args.seed = seed + i

                # 这里默认使用带对比学习的包装器。
                res = main(args, model_wrapper=ContrastiveWrapper)
                print(args)
                print(dataset, args.model_name, res)

                # 结果按“数据集-模型名-运行轮次”组织，方便整体保存。
                res_dict[f"{dataset}-{args.model_name}_run-{i}"] = res
    except Exception as e:
        print(e)
        raise e
    finally:
        # 无论中途是否报错，都会尽量把当前已经得到的结果落盘。
        print(res_dict)
        result_path = os.path.join(job_dir, f"result_{job_id}.pth")
        torch.save(res_dict, result_path)

        # 保存总结果与参数配置的简洁 txt。
        summary_path = os.path.join(job_dir, f"summary_{job_id}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"job_id: {job_id}\n")
            f.write(f"dataset: {dataset}\n")
            f.write(f"model_name: {model_name}\n")
            f.write(f"result_path: {result_path}\n")
            f.write(f"res_dict: {res_dict}\n")
            f.write("\n")
            f.write("params:\n")
            for k in [
                "seed", "batch_size", "n_gnn", "n_mlp", "max_epochs", "eval_step", "split",
                "n_data", "num_neigh", "lr", "weight_decay", "dropout", "add_SE", "device",
                "use_drug_struct", "drug_encoder_type", "drug_struct_dim", "drug_smiles_csv",
                "molformer_feat_path", "use_time_feature", "time_dim"
            ]:
                f.write(f"{k}: {getattr(args, k, None)}\n")

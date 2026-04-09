import torch
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score


def seed_everything(seed: int):
    """统一设置常见随机源，尽量保证实验可复现。"""
    import random, os
    import numpy as np
    import torch

    # Python 原生随机数。
    random.seed(seed)

    # 影响 Python 哈希相关的随机性。
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy 随机数。
    np.random.seed(seed)

    # PyTorch CPU / GPU 随机数。
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 固定 cudnn 的确定性行为。
    torch.backends.cudnn.deterministic = True

    # 保留 benchmark=True 是当前实现的一部分，这里不改逻辑，只说明它会尝试选择更快算法。
    torch.backends.cudnn.benchmark = True


def cal_auc_sperately(y_pred, y, top_k=1):
    """按类别分别计算 AUC、Precision、Recall。"""
    num_class = y.size(1)
    auc_res = []
    p_res = []
    r_res = []

    # 先按每条样本的预测分数降序排序。
    top_index = torch.argsort(y_pred, dim=1, descending=True)
    y_pred_label = torch.zeros_like(y_pred, dtype=torch.long)

    for i in range(len(top_index)):
        # 只保留 top_k 个类别作为当前样本的正预测。
        index = top_index[i][:top_k]
        y_pred_label[i][index] = 1

    # 只保留被选中的 top_k 分数，其它位置清零。
    y_pred_logits = y_pred * y_pred_label

    for i in range(num_class):
        label = y[:, i]
        pred_label = y_pred_label[:, i]
        pred_logits = y_pred_logits[:, i]

        # 全正或全负类别无法正常计算 AUC / Precision / Recall，这里统一记为 -1。
        if label.sum() == label.size(0) or label.sum() == 0:
            auc_res.append(-1)
            p_res.append(-1)
            r_res.append(-1)
        else:
            auc_res.append(roc_auc_score(label, pred_logits))
            p_res.append(precision_score(label, pred_label))
            r_res.append(recall_score(label, pred_label))

    return auc_res, p_res, r_res


def cal_sensitivity_and_specificity(y_pred, y, top_k=20):
    """按类别分别计算灵敏度和特异度。"""
    num_class = y.size(1)
    sen_res = []
    spe_res = []

    # 同样先把每条样本截断成 top_k 预测标签。
    top_index = torch.argsort(y_pred, dim=1, descending=True)
    y_pred_label = torch.zeros_like(y_pred, dtype=torch.long)
    for i in range(len(top_index)):
        index = top_index[i][:top_k]
        y_pred_label[i][index] = 1

    for i in range(num_class):
        label = y[:, i]
        pred = y_pred_label[:, i]

        # 灵敏度只在真实正样本上统计。
        sen_mask = (label == 1)

        # 特异度只在真实负样本上统计。
        spe_mask = (label == 0)

        sen_res.append(accuracy_score(label[sen_mask], pred[sen_mask]))
        spe_res.append(accuracy_score(label[spe_mask], pred[spe_mask]))

    return sen_res, spe_res


def build_save_path(args, score=0.0):
    """把核心实验配置编码进 checkpoint 文件名。"""
    run_id = getattr(args, "run_id", "")
    run_part = f"_run_{run_id}" if run_id else ""
    return (
        f"{args.dataset}_{args.model_name}_seed_{args.seed}"
        f"_n_gnn_{args.n_gnn}_n_mlp_{args.n_mlp}"
        f"{run_part}_score_{score:.4f}.ckpt"
    )

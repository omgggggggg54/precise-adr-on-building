from main import *

import datetime
import time
# main_eval.py 或 main.py 最开头（import 后）
import os, argparse, torch
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
torch.serialization.add_safe_globals([argparse.Namespace])

if __name__ == "__main__":
    # 按论文设置，依次评估 all / gender / age 三个数据子集。
    overall_job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    overall_job_dir = os.path.join("outputs", "eval_jobs", overall_job_id)
    os.makedirs(overall_job_dir, exist_ok=True)#如果目标路径的任何一级父目录不存在，都会自动逐层创建。
    overall_res_dict = {}

    for dataset in ["all", "gender", "age"]:
        for model_name in ["PreciseADR_HGT"]:
            dataset_start_time = time.time()
            start_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n===== 开始评估数据集: {dataset} | start={start_clock} =====")

            parser = argparse.ArgumentParser(description="PreciseADR")#步骤1: 创建解析器
            register_args(parser, config_file=f"config/{dataset}_HGT_config.yaml")#注册参数（定义格子）register_args 直接修改 parser 对象
            args = parse_args_and_yaml(parser)
            seed = args.seed

            if "cuda" in args.device:
                args.device = args.device if torch.cuda.is_available() else "cpu"

            res_dict = {}
            job_dir = os.path.join("outputs", dataset, "eval_jobs", overall_job_id)
            os.makedirs(job_dir, exist_ok=True)
            try:
                for i in range(1):
                    args.model_name = model_name
                    args.seed = seed + i
                    print(f"[{dataset}] run-{i} 开始 | model={args.model_name} | seed={args.seed}")

                    res = main(args, model_wrapper=ContrastiveWrapper)
                    print(args)
                    print(dataset, args.model_name, res)
                    print(f"[{dataset}] run-{i} 完成 | best_ckpt={getattr(args, 'best_ckpt_path', None)}")
                    result_key = f"{dataset}-{args.model_name}_run-{i}"
                    res_dict[result_key] = res
                    overall_res_dict[result_key] = res
            except Exception as e:
                print(e)
                raise e
            finally:
                dataset_cost = time.time() - dataset_start_time
                print(f"===== 数据集 {dataset} 评估完成 =====")
                print(f"===== 数据集 {dataset} 耗时: {dataset_cost:.1f}s =====")
                print(res_dict)
                result_path = os.path.join(job_dir, f"eval_result_{overall_job_id}.pth")
                torch.save(res_dict, result_path)

                # 保存简洁 txt，总结结果和关键参数。
                summary_path = os.path.join(job_dir, f"eval_summary_{overall_job_id}.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(f"job_id: {overall_job_id}\n")
                    f.write(f"dataset: {dataset}\n")
                    f.write(f"model_name: {model_name}\n")
                    f.write(f"result_path: {result_path}\n")
                    f.write(f"res_dict: {res_dict}\n")
                    f.write("\n")
                    f.write("params:\n")
                    # 这里必须输出「最终 args」的全部参数：
                    # args 已经经过 parse_args_and_yaml()（加载 yaml 并被命令行覆盖）得到最终值。
                    # 统一排序输出，便于不同实验之间做 diff 对比。
                    for k, v in sorted(vars(args).items()):
                        f.write(f"{k}: {v}\n")

    print("\n===== 全部数据集评估完成 =====")
    print(overall_res_dict)

    overall_result_path = os.path.join(overall_job_dir, f"all_eval_result_{overall_job_id}.pth")
    torch.save(overall_res_dict, overall_result_path)

    overall_summary_path = os.path.join(overall_job_dir, f"all_eval_summary_{overall_job_id}.txt")
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        f.write(f"job_id: {overall_job_id}\n")
        f.write(f"result_path: {overall_result_path}\n")
        f.write(f"res_dict: {overall_res_dict}\n")

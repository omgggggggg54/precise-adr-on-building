from main import *

import datetime
# main_eval.py 或 main.py 最开头（import 后）
import os, argparse, torch
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
torch.serialization.add_safe_globals([argparse.Namespace])

if __name__ == "__main__":
 # 按论文设置，依次评估 all / gender / age 三个数据子集。
    for dataset in ["all", "gender", "age"]:
        for model_name in ["PreciseADR_HGT"]:
            parser = argparse.ArgumentParser(description="PreciseADR")#步骤1: 创建解析器
            register_args(parser, config_file=f"config/{dataset}_HGT_config.yaml")#注册参数（定义格子）
            args = parse_args_and_yaml(parser)
            seed = args.seed

            if "cuda" in args.device:
                args.device = args.device if torch.cuda.is_available() else "cpu"

            res_dict = {}
            job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            job_dir = os.path.join("outputs", dataset, "eval_jobs", job_id)
            os.makedirs(job_dir, exist_ok=True)
            try:
                for i in range(1):
                    args.model_name = model_name
                    args.seed = seed + 0

                    res = main(args, model_wrapper=ContrastiveWrapper)
                    print(args)
                    print(dataset, args.model_name, res)
                    res_dict[f"{dataset}-{args.model_name}_run-{i}"] = res
            except Exception as e:
                print(e)
                raise e
            finally:
                print(res_dict)
                result_path = os.path.join(job_dir, f"eval_result_{job_id}.pth")
                torch.save(res_dict, result_path)

                # 保存简洁 txt，总结结果和关键参数。
                summary_path = os.path.join(job_dir, f"eval_summary_{job_id}.txt")
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
                        "n_data", "num_neigh", "lr", "weight_decay", "dropout", "add_SE", "device"
                    ]:
                        f.write(f"{k}: {getattr(args, k, None)}\n")

import os.path
import yaml

def register_args(parser, config_file="config/all_HGT_config.yaml"):
    # GNN related
    parser.add_argument("--model_name", type=str, default="PreciseADR_HGT")
    parser.add_argument("--add_SE",  default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=10240)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--n_mlp", type=int, default=3)
    # 按当前复现目标，默认使用 1 层图传播。
    parser.add_argument("--n_gnn", type=int, default=1)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--info_loss_weight", type=float, default=0.5)
    parser.add_argument("--loss_weight", type=list, default=None)
    parser.add_argument("--num_neigh", type=int, default=10)
    # DataLoader 并发进程数。大图场景默认 0 更稳，避免首个 batch 卡死。
    parser.add_argument("--num_workers", type=int, default=0)


    # training related
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--shuffle_attr", type=bool, default=True)
    parser.add_argument("--num_train_per_class", type=int, default=10)
    parser.add_argument("--show_training", type=bool, default=False)
    # parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    # 是否在每个 batch 前后调用 empty_cache（默认关闭，避免训练显著变慢）。
    parser.add_argument("--use_empty_cache_hook", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--in_drop", type=float, default=0.3)
    parser.add_argument("--aug_add", type=float, default=0.3)
    parser.add_argument("--edge_drop", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0, help="gamma in focal loss")

    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--patient", type=int, default=5, )
    parser.add_argument("--lr", type=float, default=5e-4, )
    parser.add_argument("--weight_decay", type=float, default=5e-5, )
    parser.add_argument("--split", type=str, default="random")
    # parser.add_argument("--split", type=str, default="by_label")
    parser.add_argument("--n_data", type=int, default=0)
    parser.add_argument("--se_min_count", type=int, default=100)
    # 药物结构编码配置。
    parser.add_argument("--use_drug_struct", type=bool, default=True)
    parser.add_argument("--drug_encoder_type", type=str, default="molformer")
    parser.add_argument("--drug_struct_dim", type=int, default=128)
    parser.add_argument("--drug_smiles_csv", type=str, default="new_data_in/refined_data/drugbank_id_smiles.csv")
    parser.add_argument("--molformer_feat_path", type=str, default="new_data_in/refined_data/drugbank_molformer_features.csv")
    # 药物共现图配置。
    parser.add_argument("--use_drug_cooccur", type=bool, default=True)
    parser.add_argument("--drug_cooccur_min_count", type=int, default=10)
    # 时序建模配置。
    parser.add_argument("--use_time_feature", type=bool, default=True)
    parser.add_argument("--time_dim", type=int, default=32)
    # 输出层标签关系图配置。
    parser.add_argument("--use_label_gnn", type=bool, default=True)
    parser.add_argument("--label_gnn_topk", type=int, default=20)
    parser.add_argument("--label_gnn_metric", type=str, default="jaccard")
    # patient 侧药物结构聚合配置。
    parser.add_argument("--use_patient_drug_agg", type=bool, default=True)
    parser.add_argument("--patient_drug_agg_type", type=str, default="mean")
    parser.add_argument("--config", type=str, default=config_file)


def parse_args_and_yaml(given_parser):
    '''优先级：硬编码默认值 → yaml文件 → 命令行参数'''

    # 仅用于获取 --config 参数位置
    args_config, _ = given_parser.parse_known_args()
    
    # 根据 yaml 配置更新默认值
    if args_config.config:
        if os.path.exists(args_config.config):
            with open(args_config.config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                given_parser.set_defaults(**cfg)
        else:
            raise RuntimeError(f"Config file {args_config.config} not exists")

    # 重新解析所有参数（应用了 yaml 的默认值后，命令行参数最终覆盖）
    args = given_parser.parse_args()

    # 保存参数快照为文本
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args

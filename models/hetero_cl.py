from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv, GATConv

from models.base import model_dict#变量import
from .base import BasicModelWrapper, Time2Vec
from .info_nce import info_nce
'''扩展模型，增加对比学习分支（InfoNCE）和对应的 Lightning 包装器（ContrastiveWrapper'''

class PreciseADR_RGCN(nn.Module):
    """带对比学习分支的异构图模型，基础卷积算子使用 GraphSAGE。"""

    def __init__(self, metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args=None):
        super(PreciseADR_RGCN, self).__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.in_dim_dict = in_dim_dict

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_info = n_info

        self.n_node = n_node
        self.in_dim = in_dim
        self.args = args
        self.n_layer = self.args.n_gnn
        self.n_gnn = args.n_gnn
        self.n_mlp = args.n_mlp if hasattr(args, "n_mlp") else 1
        self.use_time_feature = bool(getattr(args, "use_time_feature", True))
        self.time_dim = int(getattr(args, "time_dim", 32))
        self.use_drug_struct = bool(getattr(args, "use_drug_struct", True)) and int(getattr(args, "drug_struct_dim", 0)) > 0
        self.drug_struct_dim = int(getattr(args, "drug_struct_dim", 0))

        # in_lin 把统一 BOW 输入先映射到隐藏空间，供大多数节点类型共享。
        nn_layers = []
        for i in range(self.n_mlp):
            in_dim = self.in_dim if i == 0 else hid_dim
            nn_layers.append(nn.Linear(in_dim, hid_dim))
            nn_layers.append(nn.Tanh())
            nn_layers.append(nn.Dropout(args.dropout))
        self.in_lin = nn.Sequential(*nn_layers)

        # readout 负责把 patient 隐层表示映射成多标签分类 logits。
        self.readout = nn.Linear(self.hid_dim, self.out_dim)

        # SE 节点单独走一套投影层，避免与其它节点完全共享参数。
        ae_layers = []
        for i in range(self.n_mlp):
            in_dim = self.in_dim if i == 0 else hid_dim
            ae_layers.append(nn.Linear(in_dim, hid_dim))
            ae_layers.append(nn.Tanh())
            ae_layers.append(nn.Dropout(args.dropout))
        self.se_trans = nn.Sequential(*ae_layers)

        # 对比学习支路把 patient 原始输入投影到隐藏空间。
        self.cl_lin = nn.Linear(self.in_dim, self.hid_dim)

        # 构建异构卷积层，每种边类型各自拥有一套 SAGEConv。
        self.convs = nn.ModuleList()
        for _ in range(self.n_gnn):
            conv = HeteroConv({
                edge_type: SAGEConv(hid_dim, hid_dim)
                for edge_type in metadata[1]#(node_tyep,edge_type)
            })
            self.convs.append(conv)

        # edge_drop 用于训练时随机删掉部分 patient-SE 边，降低标签边过强依赖。
        self.edge_drop = nn.Dropout(args.edge_drop if hasattr(args, "edge_drop") else 0.9)

        # aug_add 用于在 patient 输入上随机加激活，构造对比学习增强视图。
        self.aug_add = nn.Dropout(args.aug_add if hasattr(args, "aug_add") else 0.1)

        # 时间编码：Time2Vec -> 投影回统一输入维度。
        self.time_encoder = Time2Vec(self.time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.in_dim),
            nn.Tanh(),
            nn.Dropout(args.dropout),
        )

        # 药物结构编码：投影后与 drug 输入相加。
        if self.use_drug_struct:
            self.drug_struct_proj = nn.Sequential(
                nn.Linear(self.drug_struct_dim, self.in_dim),
                nn.Tanh(),
                nn.Dropout(args.dropout),
            )
        else:
            self.drug_struct_proj = None

    def build_aug(self, x):
        """在输入特征上随机补 1，生成对比学习使用的增强视图。"""
        aug_x = torch.zeros_like(x)

        # 先构造全 1 掩码，再用 Dropout 产生随机保留位置。
        aug_mask = torch.ones_like(x)

        # Dropout 后小于 1 的位置表示该位置被置零，对应这里要人为补成 1 的增强位。
        aug_mask = (self.aug_add(aug_mask) < 1)
        aug_x[aug_mask] = 1
        return x + aug_x

    def _build_x_feat(self, x):
        """把输入 BOW 特征映射到隐藏空间。"""
        x = self.in_lin(x)
        return x

    def _inject_aux_features(self, x_dict, patient_time=None, drug_struct_feat=None):
        """把时间和药物结构信息注入到原始输入空间。"""
        if self.use_time_feature and patient_time is not None and "patient" in x_dict:
            time_emb = self.time_proj(self.time_encoder(patient_time))
            x_dict["patient"] = x_dict["patient"] + time_emb

        if self.use_drug_struct and self.drug_struct_proj is not None and drug_struct_feat is not None and "drug" in x_dict:
            struct_emb = self.drug_struct_proj(drug_struct_feat.float())
            x_dict["drug"] = x_dict["drug"] + struct_emb

        return x_dict

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        """前向传播，同时支持返回对比学习所需的中间隐藏向量。"""
        # 为了避免直接改动调用方传入的字典，这里先复制。
        x_dict = dict(x_dict)
        edge_index_dict = dict(edge_index_dict)
        patient_time = x_dict.pop("patient_time", None)
        drug_struct_feat = x_dict.pop("drug_struct_feat", None)

        if "info_nodes" in x_dict:
            x_dict.pop("info_nodes")
        if "attrs" in x_dict:
            x_dict.pop("attrs")

        batch_size = -1
        if "batch_size" in x_dict:
            batch_size = x_dict.pop("batch_size")

        x_dict = self._inject_aux_features(
            x_dict=x_dict,
            patient_time=patient_time,
            drug_struct_feat=drug_struct_feat,
        )

        # 对比学习分支：基于原始 patient 特征构造增强视图和辅助投影。
        x_aug = self.build_aug(x_dict["patient"][:batch_size])
        hid_aug = self.cl_lin(x_aug)
        hid_assis = self.cl_lin(x_dict["patient"][:batch_size])

        # 节点输入编码：SE 走独立投影，其它节点共享 in_lin。
        for node_type in self.node_types:
            if node_type == "SE":
                x_dict[node_type] = self.se_trans(x_dict[node_type])
            else:
                x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        # 训练时可选丢弃一部分 patient-SE 边，降低标签边泄露。
        if self.args.add_SE:
            edge_index = edge_index_dict[("patient", "p_se", "SE")]
            edge_mask = torch.ones(edge_index.size(1), device=edge_index.device)
            edge_mask = self.edge_drop(edge_mask).bool()
            edge_index = edge_index[:, edge_mask]
            edge_index_dict[("patient", "p_se", "SE")] = edge_index
            edge_index_dict[("SE", "rev_p_se", "patient")] = edge_index[[1, 0], :]

        # 图传播：按层更新 x_dict，与你说的 x_dict=tmp_dict 保持一致。
        for conv in self.convs:
            next_x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict:
                if key not in next_x_dict:
                    next_x_dict[key] = nn.Tanh()(x_dict[key])
            x_dict = next_x_dict

        hidden = x_dict["patient"][:batch_size]
        x = self.readout(hidden + hid_assis)

        if return_hidden:
            return x, hidden, hid_assis, hid_aug
        return x


class PreciseADR_HGT(PreciseADR_RGCN):
    """带对比学习分支的 HGT 版本模型。"""

    def __init__(self, metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args=None):
        self.n_gnn = args.n_gnn
        self.n_mlp = args.n_mlp
        super(PreciseADR_HGT, self).__init__(metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args)

        # 用 HGTConv 替换掉父类中的 HeteroConv + SAGEConv。
        self.convs = nn.ModuleList()
        for _ in range(self.n_gnn):
            conv = HGTConv(hid_dim, hid_dim, metadata, 1)
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        """HGT 前向传播。"""
        x_dict = dict(x_dict)
        edge_index_dict = dict(edge_index_dict)
        patient_time = x_dict.pop("patient_time", None)
        drug_struct_feat = x_dict.pop("drug_struct_feat", None)

        if "info_nodes" in x_dict:
            x_dict.pop("info_nodes")
        if "attrs" in x_dict:
            x_dict.pop("attrs")

        batch_size = -1
        if "batch_size" in x_dict:
            batch_size = x_dict.pop("batch_size")

        x_dict = self._inject_aux_features(
            x_dict=x_dict,
            patient_time=patient_time,
            drug_struct_feat=drug_struct_feat,
        )

        x_aug = self.build_aug(x_dict["patient"][:batch_size])#增强视图的输入特征，原始特征上随机补 1
        hid_aug = self.cl_lin(x_aug)
        hid_assis = self.cl_lin(x_dict["patient"][:batch_size])

        for node_type in self.node_types:
            if node_type == "SE":
                x_dict[node_type] = self.se_trans(x_dict[node_type])
            else:
                x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        if self.args.add_SE:
            edge_index = edge_index_dict[("patient", "p_se", "SE")]
            edge_mask = torch.ones(edge_index.size(1), device=edge_index.device)
            edge_mask = self.edge_drop(edge_mask).bool()
            edge_index = edge_index[:, edge_mask]
            edge_index_dict[("patient", "p_se", "SE")] = edge_index
            edge_index_dict[("SE", "rev_p_se", "patient")] = edge_index[[1, 0], :]

        for conv in self.convs:
            next_x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict:
                if key not in next_x_dict:
                    next_x_dict[key] = x_dict[key]
            x_dict = next_x_dict

        hidden = x_dict["patient"][:batch_size]
        x = self.readout(hidden + hid_assis)

        if return_hidden:
            return x, hidden, hid_assis, hid_aug#原始BOW特征的投影hid_assis|增强视图的投影hid_aug
        return x

#从base里import
#lambda意义在def 某个函数(args):return PreciseADR_RGCN(参数...)
cur_model_dict = {
    "PreciseADR_RGCN": lambda args: PreciseADR_RGCN(
        args.metadata,
        args.in_dim_dict,
        args.hid_dim,
        args.out_dim,
        n_info=args.num_info,
        n_node=args.num_node,
        in_dim=args.in_dim,
        args=args
    ),
    "PreciseADR_HGT": lambda args: PreciseADR_HGT(
        args.metadata,
        args.in_dim_dict,
        args.hid_dim,
        args.out_dim,
        n_info=args.num_info,
        n_node=args.num_node,
        in_dim=args.in_dim,
        args=args
    )
}

# 把带对比学习的模型注册进全局字典，供外部按名字实例化。
model_dict.update(cur_model_dict)
print(model_dict)


class ContrastiveWrapper(BasicModelWrapper):
    """在基础 Lightning 包装器上增加对比学习损失。"""
     # torch_geometric.data.Batch（它本身也是一个 HeteroData 对象）batch['patience']包含batch_size本次作为种子节点的patient及邻居patient 
    def common_step(self, batch: Batch, return_aug=False, return_hidden=False) -> Tuple[Tensor, Tensor]:
        """抽取 batch 中的 patient 监督部分，并执行一次模型前向。
        每个 batch 开始时，被 training_step 调用"""
        batch_size = batch['patient'].batch_size
        x_dict = {"batch_size": batch_size}

        # 将异构 batch 中每种节点的 bow_feat 整理成模型所需的 x_dict。
        for n_type in batch.node_types:
            x_dict[n_type] = batch[n_type].bow_feat
        if "time_days" in batch["patient"]:
            x_dict["patient_time"] = batch["patient"].time_days
        if "drug" in batch.node_types and "struct_feat" in batch["drug"]:
            x_dict["drug_struct_feat"] = batch["drug"].struct_feat

        # 只取种子 patient 的标签，邻居节点不参与当前 batch 的监督。标签是[num_SE节点数]的多标签二分类标签，patient节点的标签在batch['patient'].y中，邻居节点的标签不参与监督训练。
        y = batch['patient'].y[:batch_size]
        y_hat = self.model(x_dict, batch.edge_index_dict, return_hidden=return_hidden)

        if return_hidden:
            # 返回 logits 与三组隐藏向量时，同样只保留种子 patient 部分。
            #y_hat 分类 logits（预测分数）|hid_pred GNN 输出的患者隐藏表示|hid_assist 原始 BOW 特征的投影|hid_aug 增强视图的投影
            y_hat = (y_hat[0][:batch_size], y_hat[1][:batch_size], y_hat[2][:batch_size], y_hat[3][:batch_size])
        else:
            y_hat = y_hat[:batch_size]#只取作为种子节点的标签进行监督训练

        return y_hat, y

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """训练阶段同时优化监督分类损失和 InfoNCE 对比损失。
        每个训练 batch"""
        y_hat, y = self.common_step(batch, return_hidden=True)
        y_hat, hid_pred, hid_assist, hid_aug = y_hat

        # 通过滚动一个位置构造负样本，保持实现简单直接。
        hid_neg = torch.roll(hid_aug, shifts=1)

        temperature = 0.1 if not hasattr(self.args, "temperature") else self.args.temperature
        info_loss_weight = 0.5 if not hasattr(self.args, "info_loss_weight") else self.args.info_loss_weight

        # 对比学习部分只约束隐藏表示，不直接约束标签输出。
        infonce_loss = info_nce(hid_pred, hid_aug, hid_neg, temperature=temperature)

        # 总损失是监督损失与对比损失的加权和。
        loss = (1 - info_loss_weight) * self.loss_fn(y_hat, y) + info_loss_weight * infonce_loss

        # 训练阶段仍然记录监督任务的 AUC，方便观察收敛趋势。
        self.auc(y_hat, y.long())
        self.log(f'train_auc', self.auc, prog_bar=True, on_epoch=True)

        del y_hat, y
        return loss

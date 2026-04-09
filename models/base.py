"""
PreciseADR 的基础模型定义。
这里提供不带对比学习损失的模型主体，以及统一的 Lightning 包装器。
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, Dropout, LayerNorm, functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv, GATConv
from torch_geometric.typing import NodeType, EdgeType
from torchmetrics.classification import MultilabelAUROC, MultilabelPrecision, MultilabelRecall

from models.utils import RetrievalHitRate, RetrievalPrecision, RetrievalRecall, FocalLoss, RetrievalNormalizedDCG


class BasicEncoder(nn.Module):
    """一个简化的前馈编码块，结构类似 Transformer FFN + 残差归一化。"""

    def __init__(self, in_dim: int, hid_dim: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, activation=F.relu):
        super(BasicEncoder, self).__init__()

        # 两层前馈网络，输入和输出维度保持一致，中间升到 hid_dim。
        self.linear1 = Linear(in_dim, hid_dim)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hid_dim, in_dim)

        # 两次 LayerNorm 让前馈块训练更稳定。
        self.norm1 = LayerNorm(in_dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(in_dim, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        """显式重置各层参数。"""
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """先归一化，再走前馈块并做残差叠加。"""
        x = self.norm1(x)
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        """前馈子块：线性 -> 激活 -> dropout -> 线性 -> dropout。"""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class HeteroEncoder(nn.Module):
    """为不同类型节点准备独立编码器的一层封装。"""

    def __init__(self, in_dim: int, hid_dim: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, activation=F.relu, num_types=1):
        super(HeteroEncoder, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.num_types = num_types

        # 每种类型各自持有一个 BasicEncoder，不共享参数。
        self.lins = torch.nn.ModuleList([
            BasicEncoder(in_dim, hid_dim, dropout, layer_norm_eps, activation)
            for _ in range(num_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """依次重置每个类型编码器的参数。"""
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        """
        Args:
            x: 所有节点共享的大特征矩阵。
            type_vec: 每一行样本对应的类型 id。
        """
        out = x.new_empty(x.size(0), self.in_dim)
        for i, lin in enumerate(self.lins):
            # 只把当前类型的样本送入对应编码器。
            mask = type_vec == i
            out[mask] = lin(x[mask])
        return out


class PreciseADR_RGCN(nn.Module):
    """基础异构图模型，卷积算子使用 GraphSAGE。"""

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

        # 先把统一 BOW 特征投影到隐藏空间。
        nn_layers = []
        for i in range(self.n_mlp):
            in_dim = in_dim if i == 0 else hid_dim
            nn_layers.append(nn.Linear(in_dim, hid_dim))
            nn_layers.append(nn.Tanh())
            nn_layers.append(nn.Dropout(args.dropout))
        self.in_lin = nn.Sequential(*nn_layers)

        # 最终输出层负责生成每个 SE 标签的预测分数。
        self.readout = nn.Linear(self.hid_dim, self.out_dim)

        # 每层异构卷积会为每种边类型创建一套 SAGEConv。
        self.convs = nn.ModuleList()
        for _ in range(self.n_gnn):
            conv = HeteroConv({
                edge_type: SAGEConv(hid_dim, hid_dim)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def _build_x_feat(self, x):
        """把原始输入特征投影到模型隐藏空间。"""
        x = self.in_lin(x)
        return x

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        """异构图前向传播。"""
        if "info_nodes" in x_dict:
            x_dict.pop("info_nodes")

        if "attrs" in x_dict:
            x_dict.pop("attrs")

        batch_size = -1
        if "batch_size" in x_dict:
            # 取出种子节点数量，后面只对这部分 patient 做监督。
            batch_size = x_dict.pop("batch_size")

        # 所有节点类型共享同一套输入投影。
        for node_type in self.node_types:
            x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        res = [x_dict["patient"]]
        for i, conv in enumerate(self.convs):
            # 先计算当前层卷积输出。
            tmp_x_dict = conv(x_dict, edge_index_dict)

            # 没有被当前层更新到的节点类型，保留一份旧表示的非线性变换。
            for key in x_dict:
                if key not in tmp_x_dict:
                    tmp_x_dict[key] = nn.Tanh()(x_dict[key])

            # 关键修复：把本层输出回写，下一层才能基于更新后的表示继续传播。
            x_dict = tmp_x_dict
            res.append(x_dict["patient"])

        hidden = x_dict["patient"]
        x = self.readout(hidden)
        if return_hidden:
            return x, hidden
        else:
            return x


class PreciseADR_HGT(PreciseADR_RGCN):
    """基础 HGT 版本模型。"""

    def __init__(self, metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args=None):
        self.n_gnn = args.n_gnn
        self.n_mlp = args.n_mlp
        super(PreciseADR_HGT, self).__init__(metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args)

        # 用 HGTConv 替换父类中的 HeteroConv。
        self.convs = nn.ModuleList()
        for _ in range(self.n_gnn):
            conv = HGTConv(hid_dim, hid_dim, metadata, 1)
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        """HGT 前向传播。"""
        if "info_nodes" in x_dict:
            x_dict.pop("info_nodes")

        if "attrs" in x_dict:
            x_dict.pop("attrs")

        batch_size = -1
        if "batch_size" in x_dict:
            batch_size = x_dict.pop("batch_size")

        for node_type in self.node_types:
            x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        res = [x_dict["patient"]]
        att_weight = None
        for i, conv in enumerate(self.convs):
            # 如果未来要分析注意力，这里本来是一个预留位置。
            tmp_x_dict = conv(x_dict, edge_index_dict)

            # 没有更新到的节点类型直接沿用旧表示。
            for key in x_dict:
                if key not in tmp_x_dict:
                    tmp_x_dict[key] = x_dict[key]

            # 关键修复：把本层输出回写，保证多层 HGT 真正实现层间传播。
            x_dict = tmp_x_dict
            res.append(x_dict["patient"])

        hidden = x_dict["patient"]
        x = self.readout(hidden)

        if return_hidden:
            return x, hidden
        else:
            return x


model_dict = {
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
    ),
}


def model_factory(model_name, args):
    """按模型名从注册表里实例化模型。"""
    assert model_name in model_dict, f"No model:{model_name}"#条件为假,输出后面信息
    return model_dict[model_name](args)


class BasicModelWrapper(LightningModule):
    """统一封装训练、验证、测试和预测流程。"""

    def __init__(self, model_name, args=None, model_factory_func=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters()# ← 自动保存 __init__ 的所有参数

        self.model_name = model_name

        # 默认使用全局 model_dict，也支持外部传入自定义工厂。
        if model_factory_func is None:
            self.model = model_factory(model_name, args)
        else:
            self.model = model_factory_func(model_name, args)

        # 排序检索类指标，主要看前 k 个预测的命中质量。
        self.hit_1 = RetrievalHitRate(k=1, compute_on_cpu=True)
        self.hit_2 = RetrievalHitRate(k=2, compute_on_cpu=True)
        self.hit_5 = RetrievalHitRate(k=5, compute_on_cpu=True)
        self.hit_10 = RetrievalHitRate(k=10, compute_on_cpu=True)
        self.hit_20 = RetrievalHitRate(k=20, compute_on_cpu=True)
        self.hit_50 = RetrievalHitRate(k=50, compute_on_cpu=True)
        self.ndcg = RetrievalNormalizedDCG(k=20, compute_on_cpu=True)#归一化折损累积增益

        self.p_k = RetrievalPrecision(k=10, compute_on_cpu=True)#预测的前 10 个副作用中，有多少是真正发生的
        self.r_k = RetrievalRecall(k=10, compute_on_cpu=True)#真正发生的副作用中，有多少被预测在前 10 名内

        # 多标签分类核心指标。
        self.auc = MultilabelAUROC(num_labels=args.out_dim)
        self.p = MultilabelPrecision(num_labels=args.out_dim, compute_on_cpu=True)
        self.recall = MultilabelRecall(num_labels=args.out_dim, compute_on_cpu=True)

        # 根据配置选择损失函数。
        if self.args.loss == "kl":
            self.loss_fn = torch.nn.KLDivLoss(weight=self.args.loss_weight)
        elif self.args.loss == "focal":
            self.loss_fn = FocalLoss(gamma=self.args.gamma)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.args.loss_weight)

        # 下面这些列表用于在一个 epoch 内累计预测结果，epoch 结束后统一算指标。
        self.train_y = []
        self.train_y_hat = []
        self.train_ids = []

        self.val_y = []
        self.val_y_hat = []
        self.val_ids = []

        self.test_y = []
        self.test_y_hat = []
        self.test_ids = []

    def forward(#占位或备用接口
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Tensor],
            return_att: bool = False
    ) -> Dict[NodeType, Tensor]:
        """直接透传到底层模型。"""
        return self.model(x_dict, edge_index_dict)

    def common_step(self, batch: Batch, return_att=False) -> Tuple[Tensor, Tensor]:
        """抽取一个 batch 中真正参与监督的 patient 样本并前向推理。"""
        batch_size = batch['patient'].batch_size
        x_dict = {"batch_size": batch_size}

        # 整理成底层模型需要的 x_dict 结构。
        for n_type in batch.node_types:
            x_dict[n_type] = batch[n_type].bow_feat

        # 只取种子 patient 的标签，采样到的邻居不参与损失计算。
        y = batch['patient'].y[:batch_size]
        y_hat = self(x_dict, batch.edge_index_dict)[:batch_size]
        return y_hat, y

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """单个训练 step。"""
        y_hat, y = self.common_step(batch)

        if self.args.loss == "kl":
            # KLDivLoss 需要目标是概率分布，因此这里做归一化并对预测取 log_softmax。
            y = y / y.sum(dim=-1, keepdims=True)
            y_hat = F.log_softmax(y_hat, dim=-1)

        loss = self.loss_fn(y_hat, y)
        self.auc(y_hat, y.long())
        self.log(f'train_auc', self.auc, prog_bar=True, on_epoch=True)
        del y_hat, y
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        """每个训练 batch 开始前先尝试清理一下显存碎片。"""
        torch.cuda.empty_cache()
        super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每个训练 batch 结束后释放中间对象。"""
        super().on_train_batch_end(outputs, batch, batch_idx)
        del batch, outputs
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int):
        """验证阶段只累计预测结果，不在 step 内立刻算总指标。"""
        y_hat, y = self.common_step(batch)

        # 这里构造的是“每个标签分数属于哪个 patient”的索引，供部分检索指标使用。
        indexes = batch["patient"].x[:batch['patient'].batch_size].unsqueeze(0).t().repeat([1, y.size(1)]).view(-1)

        self.val_y_hat.append(y_hat.detach().cpu().clone())
        self.val_y.append(y.detach().cpu().clone())
        self.val_ids.append(indexes.detach().cpu().clone())

        del y, y_hat, indexes

    def evaluate_metrics(self, y_list, y_hat_list, index_list, mod="val", full=False):
        """把一个 epoch 内累计的预测拼起来，统一计算指标。"""
        # 当前实现里 indexes 没有真正传入检索指标，保留 None 以维持原行为。
        y = torch.cat(y_list, dim=0)
        y_hat = torch.cat(y_hat_list, dim=0)
        indexes = None

        # 先算核心 AUC。
        self.auc.reset()
        self.auc(y_hat, y.long())
        self.log(f'{mod}_auc', self.auc.compute(), prog_bar=True, on_epoch=True)

        # 再算 Top-1 命中率。
        self.hit_1(y_hat, y, indexes=indexes)
        self.log(f'{mod}_hit_1', self.hit_1.value, prog_bar=True, on_epoch=True)

        if full:
            # 测试阶段会额外输出更完整的一组排序指标。
            self.hit_2(y_hat, y, indexes=indexes)
            self.hit_5(y_hat, y, indexes=indexes)

            self.log(f'{mod}_hit_2', self.hit_2.value, prog_bar=True, on_epoch=True)
            self.log(f'{mod}_hit_5', self.hit_5.value, prog_bar=True, on_epoch=True)

            self.hit_10(y_hat, y, indexes=indexes)
            self.log(f'{mod}_hit_10', self.hit_10.value, prog_bar=True, on_epoch=True)

            self.hit_20(y_hat, y, indexes=indexes)
            self.log(f'{mod}_hit_20', self.hit_20.value, prog_bar=True, on_epoch=True)

            self.hit_50(y_hat, y, indexes=indexes)
            self.log(f'{mod}_hit_50', self.hit_50.value, prog_bar=True, on_epoch=True)

            self.ndcg(y_hat, y, indexes=indexes)
            self.log(f'{mod}_NDCG', self.ndcg.value, prog_bar=True, on_epoch=True)

            self.r_k(y_hat, y, indexes=indexes)
            self.log(f'{mod}_recall_10', self.r_k.value, prog_bar=True, on_epoch=True)

            self.p_k(y_hat, y, indexes=indexes)
            self.log(f'{mod}_precision_10', self.p_k.value, prog_bar=True, on_epoch=True)

        # 一个 epoch 算完后，立刻清空累计容器。
        y_list.clear()
        y_hat_list.clear()
        index_list.clear()

    def on_train_epoch_end(self):
        """训练 epoch 结束后的清理。"""
        self.train_y.clear(), self.train_y_hat.clear(), self.train_ids.clear()
        super().on_train_epoch_end()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        """验证 epoch 结束后统一计算验证指标。"""
        self.evaluate_metrics(self.val_y, self.val_y_hat, self.val_ids)
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        """测试 epoch 结束后输出完整指标集。"""
        self.evaluate_metrics(self.test_y, self.test_y_hat, self.test_ids, mod="test", full=True)
        super().on_test_epoch_end()

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int):
        """测试阶段累计预测结果。"""
        y_hat, y = self.common_step(batch)

        indexes = batch["patient"].x[:batch['patient'].batch_size].unsqueeze(0).t().repeat([1, y.size(1)]).view(-1)

        self.test_y_hat.append(y_hat.detach().cpu().clone())
        self.test_y.append(y.detach().cpu().clone())
        self.test_ids.append(indexes.detach().cpu().clone())

        del y, y_hat, indexes

    def configure_optimizers(self):
        """构建优化器，并按配置决定是否启用学习率调度器。"""
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.use_scheduler:
            # 原实现里先创建了 CosineAnnealingLR，随后又被 OneCycleLR 覆盖。
            opt_sche = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.args.max_epochs)
            opt_sche = torch.optim.lr_scheduler.OneCycleLR(opt, self.args.lr * 10, total_steps=self.args.max_epochs)
            return {"optimizer": opt, "lr_scheduler": opt_sche}
        else:
            return opt

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        """预测阶段返回 logits 与真实标签，便于后处理分析。"""
        y_hat, y = self.common_step(batch)
        return y_hat, y

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """手动把自定义字段一起搬到目标设备。"""
        for n_type in batch.node_types:
            batch[n_type].bow_feat = batch[n_type].bow_feat.to(device)

        batch["patient"].y = batch["patient"].y.to(device)

        # edge_index_dict 不是标准张量字段，这里手动逐项迁移。
        edge_index_dict = {k: v.to(device) for k, v in batch.edge_index_dict.items()}
        batch.edge_index_dict = edge_index_dict
        return batch

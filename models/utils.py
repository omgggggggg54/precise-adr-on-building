import numpy as np
import torch
from torch.nn import functional as F
# 兼容服务器环境：优先新路径，失败回退旧路径。
try:
    from torchmetrics.retrieval import (
        retrieval_hit_rate,
        retrieval_normalized_dcg,
        retrieval_precision,
        retrieval_recall,
    )
except Exception:
    from torchmetrics.functional import (
        retrieval_hit_rate,
        retrieval_normalized_dcg,
        retrieval_precision,
        retrieval_recall,
    )
'''辅助指标（检索 Hit Rate、NDCG、Precision/Recall）和 Focal Loss。'''

class RetrievalHitRate(object):
    def __init__(self, k=1, compute_on_cpu=True):
        self.k = k
        self.compute_on_cpu = compute_on_cpu
        self.value = 0

    def __call__(self, y_hat, y, indexes=None):
        metrics = []
        for i in range(y_hat.size(0)):
            # 使用 torchmetrics 新接口，避免 functional 入口弃用警告。
            hit_k = retrieval_hit_rate(y_hat[i], y[i], top_k=self.k)
            metrics.append(hit_k.item())

        self.value = np.mean(metrics)

    def __float__(self):
        return self.value


class RetrievalNormalizedDCG(float):
    def __init__(self, k=1, compute_on_cpu=True):
        self.k = k
        self.compute_on_cpu = compute_on_cpu
        self.value = 0

    def __call__(self, y_hat, y, indexes=None):
        metrics = []
        for i in range(y_hat.size(0)):
            hit_k = retrieval_normalized_dcg(y_hat[i], y[i], top_k=self.k)
            metrics.append(hit_k.item())

        self.value = np.mean(metrics)

    def __float__(self):
        return self.value


class RetrievalPrecision(float):
    def __init__(self, k=1, compute_on_cpu=True):
        self.k = k
        self.compute_on_cpu = compute_on_cpu
        self.value = 0

    def __call__(self, y_hat, y, indexes=None):
        metrics = []
        for i in range(y_hat.size(0)):
            hit_k = retrieval_precision(y_hat[i], y[i], top_k=self.k)
            metrics.append(hit_k.item())

        self.value = np.mean(metrics)

    def __float__(self):
        return self.value


class RetrievalRecall(float):
    def __init__(self, k=1, compute_on_cpu=True):
        self.k = k
        self.compute_on_cpu = compute_on_cpu
        self.value = 0

    def __call__(self, y_hat, y, indexes=None):
        metrics = []
        for i in range(y_hat.size(0)):
            hit_k = retrieval_recall(y_hat[i], y[i], top_k=self.k)
            metrics.append(hit_k.item())

        self.value = np.mean(metrics)

    def __float__(self):
        return self.value


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs, targets = inputs, targets
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

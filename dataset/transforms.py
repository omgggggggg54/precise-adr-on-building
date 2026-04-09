import torch
from torch_geometric.transforms import BaseTransform
'''BaseTransform，是 PyTorch Geometric 中标准的图变换组件异构图中的 patient 节点生成训练/验证/测试的掩码（mask）
，用于后续的邻居采样和模型评估。'''

class AddSEEdges(BaseTransform):
    """为训练集 patient 节点额外补充指向其真实 SE 标签的边。"""

    def forward(self, data):
        # 如果已经存在这类边，就直接返回，避免重复添加。
        if ("patient", "p_se", "SE") in data:
            return data

        # 只给训练集节点补标签边，避免把验证/测试标签泄露进图结构。
        mask = data["patient"].train_mask

        new_edges_index = []
        aim_nodes = data["patient"].x[mask].tolist()

        for node in aim_nodes:
            # y[node] 是 multi-hot 标签向量，值 >= 1 的位置就是当前病例对应的 SE。
            dst = data["patient"].y[node]
            dst = (torch.arange(len(dst))[dst.ge(1)]).tolist()

            # 同一个 patient 需要与它的所有 SE 标签分别连边。
            src = [node] * len(dst)
            new_edges_index.append(
                torch.stack([torch.LongTensor(src), torch.LongTensor(dst)])
            )

        data["patient", "p_se", "SE"].edge_index = torch.cat(new_edges_index, dim=1)
        return data


class FaersRandomNodeSplit(BaseTransform):
    """按顺序或随机方式把 patient 节点切分为 train/val/test。"""

    def __init__(self, split="in_order", num_val=0.125, num_test=0.125):
        self.num_val = num_val
        self.num_test = num_test
        self.split = split
        super(FaersRandomNodeSplit, self).__init__()

    def forward(self, data):
        # 只有 patient 节点有监督标签，因此只切分 patient。
        num_all = data["patient"].x.size(0)
        #初始化张量为false
        train_mask = torch.zeros(num_all, dtype=torch.bool)
        val_mask = torch.zeros(num_all, dtype=torch.bool)
        test_mask = torch.zeros(num_all, dtype=torch.bool)

        # 支持传比例或传绝对数量。round() 是 Python 的内置函数，用于四舍五入取整。
        num_val = round(num_all * self.num_val) if 0 < self.num_val < 1 else self.num_val
        num_test = round(num_all * self.num_test) if 0 < self.num_test < 1 else self.num_test

        assert num_test < num_all
        assert num_val < num_all

        if self.split == "in_order":
            # 按时间排序后的数据默认后面更“新”，因此末尾划给验证和测试。
            test_mask[-num_test:] = True
            val_mask[-num_test - num_val:-num_test] = True
            train_mask[:-num_test - num_val] = True
        else:
            # 随机划分时先打乱索引，再按顺序切段。randperm用于生成从 0 到 n-1 的随机排列的整数序列
            perm = torch.randperm(num_all)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:]] = True

        data["patient"].train_mask = train_mask
        data["patient"].val_mask = val_mask
        data["patient"].test_mask = test_mask
        return data


class FaersNodeSplitByTime(BaseTransform):
    """按病例日期中的年份切分训练、验证和测试集。"""

    def __init__(self, year_val="2021", year_test="2022"):
        self.year_val = year_val
        self.year_test = year_test
        super(FaersNodeSplitByTime, self).__init__()

    def forward(self, data):
        # date 是字符串列表，直接通过包含关系匹配年份。
        date = data["patient"].date
        val_mask = torch.BoolTensor([self.year_val in each for each in date])
        test_mask = torch.BoolTensor([self.year_test in each for each in date])

        assert val_mask.sum() > 0
        assert test_mask.sum() > 0

        # 除验证和测试之外的样本都作为训练集。
        train_mask = ~(torch.logical_or(val_mask, test_mask))

        data["patient"].train_mask = train_mask
        data["patient"].val_mask = val_mask
        data["patient"].test_mask = test_mask
        return data


class FaersNodeSplitByRegion(BaseTransform):
    """按地区划分：指定国家做训练，其余国家做测试。"""

    def __init__(self, train_region="US", val_rate=0.25):
        self.train_region = train_region
        self.val_rate = val_rate
        super(FaersNodeSplitByRegion, self).__init__()

    def forward(self, data):
        country = data["patient"].country

        # 先按国家把训练候选和测试集分开。
        train_mask = torch.BoolTensor([self.train_region == each for each in country])
        assert train_mask.sum() > 0, f"No {self.train_region} data"

        test_mask = ~train_mask
        assert test_mask.sum() > 0, f"Only {self.train_region} data"

        num_train = train_mask.sum().item()
        num_val = round(num_train * self.val_rate)

        # 当前实现直接取训练区域最前面的 num_val 个样本作为验证集。
        train_indices = torch.nonzero(train_mask).view(-1)
        val_indices = train_indices[:num_val]
        train_indices = train_indices[num_val:]

        train_mask = torch.zeros_like(train_mask)
        train_mask[train_indices] = True

        val_mask = torch.zeros_like(train_mask)
        val_mask[val_indices] = True

        data["patient"].train_mask = train_mask
        data["patient"].val_mask = val_mask
        data["patient"].test_mask = test_mask
        return data


class FaersNodeSplitByLabel(FaersRandomNodeSplit):
    """
    按标签做切分：
    1. 每个类别先抽取固定数量样本进入训练集。
    2. 剩余样本中取一部分做验证。
    3. 其余样本进入测试集。
    """

    def __init__(self, split="in_order", num_train_per_class=10, num_val=0.25):
        self.split = split
        self.num_val = num_val
        self.num_train_per_class = num_train_per_class
        super(FaersNodeSplitByLabel, self).__init__()

    def forward(self, data):
        num_faers = data["patient"].x.size(0)

        train_mask = torch.zeros(num_faers, dtype=torch.bool)
        val_mask = torch.zeros(num_faers, dtype=torch.bool)
        test_mask = torch.zeros(num_faers, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round(num_faers * self.num_val)
        else:
            num_val = self.num_val

        assert num_val < num_faers

        y = data["patient"].y
        num_classes = y.size(1)

        # 对每个标签单独采样，保证训练集至少覆盖到更多类别。
        for c in range(num_classes):
            idx = (y[:, c] == 1).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))]
            idx = idx[:self.num_train_per_class]
            train_mask[idx] = True

        # 剩余样本再随机划分为验证和测试。
        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        val_mask[remaining[:num_val]] = True
        test_mask[remaining[num_val:]] = True

        data["patient"].train_mask = train_mask
        data["patient"].val_mask = val_mask
        data["patient"].test_mask = test_mask

        # seen_mask 保留训练集标签覆盖范围，便于后续分析“见过/没见过”的类别。
        data["patient"].seen_mask = train_mask
        return data

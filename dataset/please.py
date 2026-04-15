import glob
import os
import os.path as osp
import pickle
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from easydict import EasyDict as edict
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from dataset.mol_features import build_drug_struct_matrix
from dataset.transforms import *

home_path = os.getenv("HOME")
base_path = osp.dirname(__file__)
# ============================================================================
# 异构图节点特征说明
# ============================================================================
# 图包含 4 种节点类型：
#
# 1. patient（患者）
#    - x: 节点索引 (0 ~ N-1)
#    - bow_feat: BOW 向量，形状 (N, in_dim)。其中值为 1 的位置表示：
#        * 前 47 维：个人信息桶（性别/体重/年龄离散编码）
#        * 接着 num_indications 维：适应症 one-hot
#        * 接着 num_drugs 维：药物 one-hot
#        * 最后 num_SE 维：副作用（本节点全 0）
#    - y: 多标签副作用标签，形状 (N, num_SE)，multi-hot
#    - num_info: 47（个人信息特征长度）
#    - country / date: 用于按区域/时间切分
#
# 2. indication（适应症）
#    - x: 节点索引
#    - bow_feat: one-hot，位置 = 47 + indication_id
#
# 3. drug（药物）
#    - x: 节点索引
#    - bow_feat: one-hot，位置 = 47 + num_indications + drug_id
#
# 4. SE（副作用）
#    - x: 节点索引
#    - bow_feat: one-hot，位置 = 47 + num_indications + num_drugs + se_id
#
# 模型使用时，从 batch.node_types 收集各节点的 bow_feat 构成 x_dict，
# 然后分别投影（SE 用独立投影层，其余共享）再通过异构卷积更新表示。
# ============================================================================

class InfoMapping(object):
    # 个人信息特征总长度：
    # 1 个 unknown 占位 + 3 个性别桶 + 30 个体重桶 + 13 个年龄桶。
    num_elements = 1 + 3 + 30 + 13

    def rev_map(self, i):
        """把个人信息特征索引还原成可读文本。"""
        if i == 0 or i == 1:
            # 0 表示未知值占位，1 表示未知性别。
            return "unknown"
        elif i == 2:
            return "male"
        elif i == 3:
            return "female"
        elif 3 < i < (30 + 3 + 1):
            # 体重按 10kg 分桶。
            return f"weight:{10 * (i - 4)}"
        elif self.num_elements > i >= (30 + 3 + 1):
            # 年龄按 10 岁分桶。
            return f"age:{10 * (i - 30 - 3 - 1)}"
        else:
            raise RuntimeError(f"Wrong value:{i}")

    def map_gender(self, gender):
        """把性别值映射到特征索引。"""
        # 原始值通常是 0/1，这里整体右移 1 给 unknown 预留位置。
        return int(gender) + 1

    def map_weight(self, weight):
        """把体重映射到 10kg 粒度的离散桶。"""
        new_weight = int(float(weight))

        # 超出合理范围的体重统一回落到 unknown 桶。
        if new_weight > 300:
            new_weight = 0
        if new_weight < 0:
            new_weight = 0

        # 前面有 unknown 和 gender 的偏移，所以要额外加 4。
        return new_weight // 10 + 3 + 1

    def map_age(self, age):
        """把年龄映射到 10 岁粒度的离散桶。"""
        new_age = int(float(age))

        # 无效年龄统一映射到 unknown 桶。
        if new_age > 120 or new_age < 0:
            new_age = -1

        # 这里依次跳过 unknown、gender、weight 这三段空间。
        return new_age // 10 + 1 + (3 + 30 + 1)


class PLEASESource(InMemoryDataset):
    """PLEASE 数据集的图构建入口，负责把原始病例记录转成异构图。"""

    def __init__(
            self,
            root: str = f"{base_path}/",#数据集根目录路径
            n_data=0,#要使用的病例数量，0 表示使用全部。
            se_type="all",
            use_processed=True,#是否使用缓存图，False 则强制删除已处理文件。
            transform=T.RandomNodeSplit(num_val=0.125, num_test=0.125),#变换（默认随机节点划分）。
            args=None,
    ):
        self.args = args if args is not None else edict()
        self.se_type = se_type
        self.count = 0#（当前处理进度）
        project_root = osp.dirname(base_path)
        default_smiles_csv = osp.join(project_root, "new_data_in", "refined_data", "drugbank_id_smiles.csv")

        # 药物结构编码配置。
        self.use_drug_struct = bool(getattr(self.args, "use_drug_struct", True))
        self.drug_encoder_type = str(getattr(self.args, "drug_encoder_type", "hybrid"))
        self.drug_struct_dim = int(getattr(self.args, "drug_struct_dim", 128))
        self.drug_smiles_csv = str(getattr(self.args, "drug_smiles_csv", default_smiles_csv))
        self.molformer_feat_path = str(getattr(self.args, "molformer_feat_path", ""))
        # 时间特征开关（数据里始终会生成 time_days，模型侧按开关使用）。
        self.use_time_feature = bool(getattr(self.args, "use_time_feature", True))

        # n_data=0 时读取全部样本，否则只保留最近的 n_data 条。
        self.n_data = n_data if n_data > 0 else len(pickle.load(open(f"{root}/PLEASE-US-{self.se_type}.pk", 'rb')))
        self.root = root

        # 个人信息映射器在整个数据集中共享。
        self.info_mapping = InfoMapping()

        if not use_processed:
            # 强制重建时，先删除当前配置对应的 processed 文件。
            # 这些文件位于 dataset/processed/*.pt，只是处理后的图数据缓存，不是模型权重。
            # 删除后可回收磁盘空间；下次训练会自动重新处理并重建缓存
            # 文件后缀是 ".pt"，这里去掉后缀再拼 *，避免误删成 hybrid_12* 这种错误模式。
            filepath = os.path.join(root, "processed", self.processed_file_names[:-3] + "*")
            print(filepath)
            for f in glob.glob(filepath):#返回所有匹配通配符的文件列表。
                os.remove(f)

        super(PLEASESource, self).__init__(root, transform=transform)##如果缓存存在且 use_processed=True，直接加载缓存//如果缓存不存在或需要重新处理，则调用 self.process() 方法生成数据。

        # InMemoryDataset 初始化后会自动把 processed 文件读进来。slices用不上
        # PyTorch 2.6 默认 weights_only=True，会拦截 HeteroData 反序列化，这里显式关闭。
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        #self.data 是一个 HeteroData 对象，包含了整个异构图,data["node_type"] 来获取该节点类型的所有属性
        self.num_feat = self.data["patient"].bow_feat.size(1)#患者节点的 BoW 特征维度，即全局特征空间的维度（人口学 + 适应症数 + 药物数 + 副作用数）
        self.num_se = self.data["patient"].y.size(1)#副作用类别的总数，即多标签分类的输出维度

        # 收集索引映射，方便后续做特征解释。
        self.collect_maps()

    def collect_maps(self):
        """整理统一 BOW 特征索引到真实语义的映射。
        global_map存各个特征(编号)->真实意义(name)"""
        self.num_infos = self.info_mapping.num_elements

        # global_map 负责从全局列索引反查具体特征含义。
        self.global_map = {}

        self.i_map = self.data["patient"].i_map
        self.se_map = self.data["patient"].se_map
        self.d_map = self.data["patient"].d_map

        # 反向映射方便从整数 id 找回原始实体名。
        self.d_map_rev = {v: k for k, v in self.d_map.items()}
        self.i_map_rev = {v: k for k, v in self.i_map.items()}
        self.se_map_rev = {v: k for k, v in self.se_map.items()}

        # 先填个人信息区间。
        for i in range(self.info_mapping.num_elements):
            self.global_map[i] = self.info_mapping.rev_map(i)

        # indication 特征紧跟在个人信息之后。
        offset = self.info_mapping.num_elements
        for i, i_name in enumerate(self.i_map):
            self.global_map[i + offset] = i_name

        # drug 特征继续顺延。
        offset += len(self.i_map)
        for i, d_name in enumerate(self.d_map):
            self.global_map[i + offset] = d_name
     
    @property
    def processed_file_names(self) -> str:
        """processed 文件名由副作用类型和样本数共同决定。
        属性会在每次实例化 PLEASESource 时被触发"""
        #父类 InMemoryDataset 会根据这个属性自动生成 self.processed_paths
        #self.processed_dir = os.path.join(self.root, 'processed')
        return (
            f'PLEASE_{self.se_type}_v3_{self.n_data}_'
            f'{int(self.use_drug_struct)}_{self.drug_encoder_type}_{self.drug_struct_dim}.pt'
        )

    @property
    def raw_file_names(self) -> List[str]:
        """原始数据文件名。_load_data() 使用了 raw_file_names"""
        return [f'PLEASE-US-{self.se_type}.pk']

    def _build_person_graph(self, record):
        """把单条病例转成 patient 节点、边和标签所需的中间结果。"""
        # 默认不刷屏；仅在 show_training=True 时输出粗粒度进度。
        if bool(getattr(self.args, "show_training", False)) and self.count % 10000 == 0:
            print(f"{self.count}/{self.n_data}")

        infos = []
        attrs = []
        p_i_edge_index = [[], []]
        p_d_edge_index = [[], []]

        # 1. 取出集合类信息并去重。
        indications = set(record["indications"])
        drugs = set(record["drugs"])
        se = set(record["SE"])

        # 这两个字段不参与当前 BOW 编码，但后续切分会用到。
        country = record["country"]
        receipt_date = record["receipt_date"]

        # 2. 把个人信息编码成离散索引。
        gender = record["gender"]
        age = record["age"]
        weight = record["weight"]
        infos.append(self.info_mapping.map_gender(gender))
        infos.append(self.info_mapping.map_weight(weight))
        infos.append(self.info_mapping.map_age(age))

        # 3. 构建 patient -> indication 边。
        for i in indications:
            p_i_edge_index[0].append(self.count)
            p_i_edge_index[1].append(self.i_map[i])

        # 4. 构建 patient -> drug 边。
        for d in drugs:
            p_d_edge_index[0].append(self.count)
            p_d_edge_index[1].append(self.d_map[d])

        # attrs 记录 patient BOW 特征中哪些列需要置 1。
        # 这里保留原实现，只把前两个个人信息索引写进 attrs。
        attrs.extend(infos[:-1])

        # indication / drug 节点的 id 需要加上各自前缀偏移，才能映射到统一特征空间。
        attrs.extend(np.array(p_i_edge_index[1]) + self.info_mapping.num_elements)
        attrs.extend(np.array(p_d_edge_index[1]) + self.info_mapping.num_elements + len(self.i_map))

        # 5. 把原始 SE 编码映射成连续标签 id。
        se_list = list(se)
        new_se_list = []
        for each in se_list:
            new_se_list.append(self.se_map[each])

        self.count += 1
        return (
            torch.LongTensor(infos).unsqueeze(0),
            torch.LongTensor(p_i_edge_index),
            torch.LongTensor(p_d_edge_index),
            new_se_list,
            attrs,
            country,
            receipt_date
        )

    @staticmethod
    def _date_to_day_index(date_value):
        """把日期转成天粒度序号，便于后续做时序编码。"""
        try:
            # 统一到 day 精度，避免时分秒噪音。
            day_val = np.datetime64(str(date_value), "D").astype("int64")
            return int(day_val)
        except Exception:
            return 0

    def build_base_graph(self):
        """构建不依赖 patient 的基础异构图骨架。"""
        num_se = len(self.se_map)
        num_indication = len(self.i_map)
        num_drug = len(self.d_map)

        # 统一输入维度 = 个人信息 + indication + drug + SE。
        self.in_dim = self.info_mapping.num_elements + num_se + num_indication + num_drug

        # x 存实体顺序 id，bow_feat 才是真正给模型用的输入特征。
        self.hetero_graph["SE"].x = torch.arange(num_se)
        self.hetero_graph["indication"].x = torch.arange(num_indication)
        self.hetero_graph["drug"].x = torch.arange(num_drug)

        # indication 的 one-hot 区间从个人信息之后开始。
        self.hetero_graph["indication"].bow_feat = one_hot(
            torch.arange(num_indication) + self.info_mapping.num_elements,
            num_classes=self.in_dim
        ).float().to_sparse()

        # drug 的 one-hot 区间再往后顺延 num_indication 个位置。
        self.hetero_graph["drug"].bow_feat = one_hot(
            torch.arange(num_drug) + self.info_mapping.num_elements + num_indication,
            num_classes=self.in_dim
        ).float().to_sparse()

        # 药物结构编码：把 DrugBank ID 对应的分子信息编码成定长向量。
        if self.use_drug_struct:
            ordered_drug_ids = [None] * num_drug
            for drug_id, idx in self.d_map.items():
                ordered_drug_ids[idx] = drug_id

            struct_feat, struct_stat = build_drug_struct_matrix(
                ordered_drug_ids=ordered_drug_ids,
                smiles_csv_path=self.drug_smiles_csv,
                encoder_type=self.drug_encoder_type,
                struct_dim=self.drug_struct_dim,
                molformer_feat_path=self.molformer_feat_path,
            )
            self.hetero_graph["drug"].struct_feat = torch.from_numpy(struct_feat).float()
            print(
                "[DrugStruct] mode:",
                self.drug_encoder_type,
                "| dim:",
                self.drug_struct_dim,
                "| total:",
                struct_stat["total_drug"],
                "| smiles_hit:",
                struct_stat["hit_smiles"],
                "| molformer_hit:",
                struct_stat["hit_molformer"],
                "| rdkit:",
                struct_stat["rdkit_enabled"],
            )

        # SE 的 one-hot 区间位于统一特征空间的最后一段。
        self.hetero_graph["SE"].bow_feat = one_hot(
            torch.arange(num_se) + self.info_mapping.num_elements + num_indication + num_drug,
            num_classes=self.in_dim
        ).float().to_sparse()

    def _load_data(self):
        """读取原始 pickle 数据，并按 receipt_date 升序排序。"""
        filename = f"{self.root}/{self.raw_file_names[0]}"
        assert osp.exists(filename), f"{filename} doesn't exist in path {self.root}."

        all_pd = pickle.load(open(filename, 'rb'))
        all_pd = all_pd.sort_values(by="receipt_date")

        assert len(all_pd) > 0, "No data!!!"

        # 只保留时间上最新的 n_data 条样本。
        if self.n_data > 0:
            all_pd = all_pd.tail(self.n_data)
        else:
            self.n_data = len(all_pd)

        self.all_pd = all_pd
        return all_pd

    def process(self):
        """InMemoryDataset 预处理入口。"""
        all_pd = self._load_data()
        if self.n_data == 0:
            self.n_data = len(all_pd)

        # 先建立 SE / indication / drug 的全局映射。
        print("阶段 1/3：构建实体映射")
        self.build_map(all_pd)

        print("阶段 2/3：构建 patient 图")
        print("data size:", len(all_pd))

        self.hetero_graph = HeteroData()
        self.build_base_graph()
        self.extract_patient_graph(all_pd)

        # 最终把完整异构图保存到 processed 文件。
        print("阶段 3/3：保存 processed 图文件")
        torch.save(self.collate([self.hetero_graph]), self.processed_paths[0])
        #可以存储多个图,期望接收list
    def extract_patient_graph(self, all_pd):
        """遍历所有病例，补齐 patient 节点的特征、边、标签和附加属性。"""
        count = 0

        se_id_list = []
        all_p_i_edge_index = []
        all_p_d_edge_index = []
        all_y = []
        all_feats = []
        all_country = []
        all_date = []
        all_day_index = []

        # 逐条病例构图，并显示明确进度条。
        for _, row in tqdm(
            all_pd.iterrows(),
            total=len(all_pd),
            desc="2.1/3 病例转图",
            dynamic_ncols=True
        ):
            info_nodes, p_i_edge_index, p_d_edge_index, se_id, attrs, country, receipt_date = self._build_person_graph(row)
            all_p_i_edge_index.append(p_i_edge_index)
            all_p_d_edge_index.append(p_d_edge_index)
            se_id_list.append(se_id)

            # patient 的监督标签是 multi-hot，因为一条记录可能对应多个 SE。
            y = torch.zeros(1, len(self.se_map))
            y[0, se_id] = 1
            all_y.append(y)

            all_feats.append(attrs)
            all_country.append(country)
            all_date.append(receipt_date)
            all_day_index.append(self._date_to_day_index(receipt_date))
            count += 1

        # 把“激活列索引列表”恢复成真正的 BOW 向量。
        all_x = []
        for attr in tqdm(all_feats, total=len(all_feats), desc="2.2/3 组装 patient 特征", dynamic_ncols=True):
            x = torch.zeros([self.in_dim])
            x[attr] = 1
            all_x.append(x)
        x = torch.stack(all_x, dim=0)

        self.hetero_graph["patient"].x = torch.arange(count)
        self.hetero_graph["patient"].num_info = self.info_mapping.num_elements
        self.hetero_graph["patient", "p_i", "indication"].edge_index = torch.cat(all_p_i_edge_index, dim=1)
        self.hetero_graph["patient", "p_d", "drug"].edge_index = torch.cat(all_p_d_edge_index, dim=1)
        self.hetero_graph["patient"].y = torch.cat(all_y, dim=0).to_sparse()
        self.hetero_graph["patient"].bow_feat = x.to_sparse()

        # 地区和时间保留下来，给按地区/时间切分的数据增强使用。
        self.hetero_graph["patient"].country = all_country
        self.hetero_graph["patient"].date = all_date
        # 归一化后的时间特征，供模型做时序编码。
        day_arr = np.array(all_day_index, dtype=np.float32)
        if day_arr.size == 0:
            time_days = np.zeros((count, 1), dtype=np.float32)
        else:
            min_day = float(day_arr.min())
            span = max(float(day_arr.max()) - min_day, 1.0)
            time_days = ((day_arr - min_day) / span).reshape(-1, 1)
        self.hetero_graph["patient"].time_days = torch.from_numpy(time_days).float()

        # 同时把映射表挂到 patient 上，方便外部统一读取。
        self.hetero_graph["patient"].se_map = self.se_map
        self.hetero_graph["patient"].d_map = self.d_map
        self.hetero_graph["patient"].i_map = self.i_map

    def build_map(self, all_pd):
        """扫描全量数据，建立实体名称到连续 id 的映射。"""
        self.se_map = OrderedDict()
        self.i_map = OrderedDict()
        self.d_map = OrderedDict()
        # 单趟扫描建立三类映射，并显示进度。
        for _, row in tqdm(
            all_pd.iterrows(),
            total=len(all_pd),
            desc="1/3 扫描 SE/indication/drug",
            dynamic_ncols=True
        ):
            for each in sorted(set(row["SE"])):
                if each not in self.se_map:
                    self.se_map[each] = len(self.se_map)
            for each in sorted(set(row["indications"])):
                if each not in self.i_map:
                    self.i_map[each] = len(self.i_map)
            for each in sorted(set(row["drugs"])):
                if each not in self.d_map:
                    self.d_map[each] = len(self.d_map)


class DataModule(LightningDataModule):
    """Lightning 数据模块，负责数据准备、切分和邻居采样。"""

    def __init__(
            self,
            n_data=10000,
            n_layer=2,
            batch_size=10240,
            add_SE=False,
            split="in_order",
            to_homo=False,
            filtered_SE=None,
            # 默认优先复用已经处理好的图缓存，和原版行为保持一致。
            use_processed=True,
            se_type="all",
            args=None,
    ):
        super().__init__()
        self.use_processed = use_processed
        self.args = args

        if self.args is None:
            self.args = edict()#纯字典

        if hasattr(args, "filtered_SE"):
            self.filtered_SE = args.filtered_SE
        else:
            self.filtered_SE = filtered_SE

        self.dataset = None
        self.setup_over = False
        self.num_neigh = args.num_neigh if hasattr(args, "num_neigh") else 10
        self.se_type = args.se_type if hasattr(args, "se_type") else se_type

        self.to_homo = to_homo
        self.n_data = n_data
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.add_SE = add_SE
        self.split = split
        self.num_train_per_class = self.args.num_train_per_class if hasattr(self.args, "num_train_per_class") else 10
        self.seed = self.args.seed if hasattr(self.args, "seed") else 123

        # 初始化阶段就加载一次，保证外部立刻能读到 data 和索引边界。
        self.load()
        self.setup_over = True

    def load(self):
        """加载数据，并把稀疏特征切换成训练更常用的 dense 表示。"""
        self.setup()

        if self.to_homo:
            self.data.bow_feat = self.data.bow_feat.to_sparse()
            self.data.y = self.data.y.to_sparse()
        else:
            for nt in self.data.node_types:
                self.data[nt].bow_feat = self.data[nt].bow_feat.to_sparse()
            self.data["patient"].y = self.data["patient"].y.to_sparse()

        if self.to_homo:
            # 同构图模式直接处理整体特征和标签。
            self.data.bow_feat = self.data.bow_feat.to_dense()
            self.data.y = self.data.y.to_dense()
        else:
            n_feat = self.data["indication"].bow_feat.size(1)
            n_i = self.data["indication"].bow_feat.size(0)
            n_d = self.data["drug"].bow_feat.size(0)
            n_se = self.data["SE"].bow_feat.size(0)

            # 通过特征段长度反推出 indication / drug 在统一 BOW 中的位置范围。
            self.i_start = n_feat - n_se - n_d - n_i
            self.i_end = n_feat - n_se - n_d
            self.d_start = n_feat - n_se - n_d
            self.d_end = n_feat - n_se

            for nt in self.data.node_types:
                self.data[nt].bow_feat = self.data[nt].bow_feat.to_dense()
            self.data["patient"].y = self.data["patient"].y.to_dense()

    def to_hetero_data(self):
        """保持异构图结构不变，并记录必要元信息。"""
        assert self.data is not None
        num_info = self.data["patient"].num_info
        self.node_types, self.edge_types = self.data.metadata()
        y = self.data['patient'].y.clone()
        self.n_se = y.size(1)
        self.data["patient"].num_info = num_info
        return self.data

    def to_homo_data(self):
        """把异构图转换成同构图，并补齐统一字段。"""
        assert self.data is not None
        num_info = self.data["patient"].num_info
        node_types, edge_types = self.data.metadata()
        y = self.data['patient'].y.clone()
        n_se = y.size(1)
        old_bow_size = self.data["patient"].bow_feat.size(1)

        for nt in node_types:
            data_size = self.data[nt].x.size(0)

            if nt != "patient":
                # 非 patient 节点没有真实标签，这里补零只是为了统一字段结构。
                self.data[nt].y = torch.zeros([data_size, n_se])
                self.data[nt].train_mask = torch.zeros([data_size], dtype=torch.bool)
                self.data[nt].val_mask = torch.zeros([data_size], dtype=torch.bool)
                self.data[nt].test_mask = torch.zeros([data_size], dtype=torch.bool)

            if nt != "SE":
                # 非 SE 节点在扩展后的 SE 标签区间全部填 0。
                self.data[nt].bow_feat = torch.cat([self.data[nt].bow_feat, torch.zeros([data_size, n_se])], dim=-1)

        n_se = self.data["SE"].bow_feat.size(0)

        # SE 节点单独补一个单位阵，让每个 SE 节点在扩展后的尾部区间有唯一编码。
        self.data["SE"].bow_feat = torch.cat([torch.zeros([n_se, old_bow_size]), torch.eye(n_se)], dim=-1)
        print(self.data)

        homo_data = self.data.to_homogeneous()
        homo_data.num_info = num_info
        return homo_data

    def save_labels(self):
        """保存当前 patient 节点的 train/val/test mask。"""
        train_mask, val_mask, test_mask = (
            self.data["patient"].train_mask,
            self.data["patient"].val_mask,
            self.data["patient"].test_mask
        )
        f_name = f"{self.split}_{self.n_data}_label_mask.pth"
        torch.save([train_mask, val_mask, test_mask], f_name)

    def get_data_info(self):
        """整理图解释和训练阶段会用到的边界信息。"""
        # country/date 主要给特殊切分逻辑用，常规训练里删掉可以减少无关负担。
        del self.dataset.data["patient"].country
        del self.dataset.data["patient"].date

        self.i_start = self.dataset.info_mapping.num_elements
        self.i_end = self.i_start + len(self.dataset.i_map)
        self.d_start = self.i_end
        self.d_end = self.i_end + len(self.dataset.d_map)

        self.dataset.data["patient"].y = self.dataset.data["patient"].y.to_dense()
        for n_type in self.dataset.data.node_types:
            self.dataset.data[n_type].bow_feat = self.dataset.data[n_type].bow_feat.to_dense()
        self.in_dim = self.dataset.data["patient"].bow_feat.size(1)

        # 这几个映射字典很大，训练采样阶段不需要，删除可减轻 DataLoader 负担。
        if "se_map" in self.dataset.data["patient"]:
            del self.dataset.data["patient"].se_map
        if "d_map" in self.dataset.data["patient"]:
            del self.dataset.data["patient"].d_map
        if "i_map" in self.dataset.data["patient"]:
            del self.dataset.data["patient"].i_map

    def load_data(self):
        """按当前配置实例化 PLEASESource。"""
        if self.dataset is None:
            self.dataset = PLEASESource(
                n_data=self.n_data,
                se_type=self.se_type,
                use_processed=self.use_processed,
                transform=self.transform,
                args=self.args,
            )
            self.get_data_info()
        return self.dataset

    def prepare_data(self):
        """Lightning 约定接口，用于提前触发数据准备。"""
        return self.load_data()

    def setup(self, stage: Optional[str] = None):
        """构建 transform、加载数据，并按需求转成异构图或同构图。"""
        if not self.setup_over:
            transform_list = []

            # 1. 先定义标签切分策略。
            if self.split == "in_order":
                transform_list.append(FaersRandomNodeSplit())
            elif self.split == "by_label":
                transform_list.append(FaersNodeSplitByLabel(num_train_per_class=self.num_train_per_class))
            else:
                transform_list.append(T.RandomNodeSplit(num_val=0.125, num_test=0.125))

            # 2. 可选：把训练标签显式加入图结构。
            if self.add_SE:
                transform_list.append(AddSEEdges())

            # 3. 最后补反向边，让消息传递双向可达。
            transform_list.append(T.ToUndirected(merge=False))

            self.transform = T.Compose(transform_list)

            # transform 会在访问 dataset[0] 时真正生效。
            self.dataset = self.load_data()
            self.data = self.dataset[0]

            if self.to_homo:
                self.data = self.to_homo_data()
            else:
                self.data = self.to_hetero_data()

            if not self.to_homo:
                # 只打印精简后的元信息，避免超大对象打印拖慢日志。
                self.metadata = self.data.metadata()
                print("node_types:", self.metadata[0])
                print("edge_types:", self.metadata[1])
            print("Validation:", self.data.validate())
            self.setup_over = True

    def dataloader(self, mask: Tensor, shuffle: bool, num_workers: int = None, mode="train"):
        """根据 patient 掩码构建邻居采样 dataloader。"""
        batch_size = self.batch_size
        if num_workers is None:
            num_workers = int(getattr(self.args, "num_workers", 0))
        if num_workers < 0:
            num_workers = 0

        if self.to_homo:
            dataloader = NeighborLoader(
                self.data,
                num_neighbors=[self.num_neigh] * self.n_layer,
                input_nodes=mask,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                persistent_workers=num_workers > 0
            )
        else:
            # 异构图模式下，需要明确告诉采样器监督目标是 patient 节点。
            dataloader = NeighborLoader(
                self.data,#HeteroData 实例
                num_neighbors=[self.num_neigh] * self.n_layer,#生成一个列表，例如 [10, 10] 表示第一层采 10 个邻居，第二层再采 10 个邻居。
                input_nodes=('patient', mask),#指定采样的起始节点集合。
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                persistent_workers=num_workers > 0#是否保持 worker 进程在 epoch 之间存活。
            )

            # 手动保留 input_nodes，方便调试时直接查看。
            dataloader.input_nodes = ('patient', mask)

        dataloader.num_neighbors = [self.num_neigh] * self.n_layer
        if not hasattr(self, "_loader_info_printed"):
            print(
                f"[DataLoader] mode={mode} | batch_size={batch_size} | num_workers={num_workers} | "
                f"n_layer={self.n_layer} | num_neigh={self.num_neigh}"
            )
            self._loader_info_printed = True
        return dataloader

    def train_dataloader(self):
        """训练集采样器。trainer.fit每个epoch调用"""
        if self.to_homo:
            return self.dataloader(self.data.train_mask, shuffle=True)
        else:
            return self.dataloader(self.data['patient'].train_mask, shuffle=True)

    def val_dataloader(self):
        """验证集采样器。trainer.validate每个epoch调用"""
        if self.to_homo:
            return self.dataloader(self.data.val_mask, shuffle=False)
        else:
            return self.dataloader(self.data['patient'].val_mask, shuffle=False)

    def test_dataloader(self):
        """测试集采样器。trainer.test每个epoch调用"""
        if self.to_homo:
            return self.dataloader(self.data.test_mask, shuffle=False)
        else:
            return self.dataloader(self.data['patient'].test_mask, shuffle=False)


if __name__ == '__main__':
    # 手动触发一次重建流程，便于离线检查数据处理是否正常。
    for se_type in ["all", "gender", "age"]:
        dataset = PLEASESource(n_data=0, use_processed=False, se_type=se_type)
        DataModule(n_data=0, use_processed=True, se_type=se_type)

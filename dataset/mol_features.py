import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

# Morgan 指纹固定位数，2048 位减少哈希碰撞，再投影到 dim
_FP_NBITS = 2048

# 物化描述符 clip 先验范围 (lo, hi)，顺序与 encode_physchem 保持一致
_PHYSCHEM_CLIPS = [
    (0.0, 1500.0),   # MolWt
    (-5.0, 10.0),    # MolLogP
    (0.0, 300.0),    # TPSA
    (0.0, 20.0),     # NumHDonors
    (0.0, 20.0),     # NumHAcceptors
    (0.0, 30.0),     # NumRotatableBonds
    (0.0, 10.0),     # NumAromaticRings
    (0.0, 15.0),     # RingCount
    (0.0, 1.0),      # FractionCSP3
    (0.0, 100.0),    # NumHeavyAtoms
    (0.0, 30.0),     # NumHeteroatoms
    (-1.0, 1.0),     # MaxPartialCharge
    (-1.0, 1.0),     # MinPartialCharge
]


def _stable_hash(token: str) -> int:
    """稳定哈希，避免 Python 内置 hash 的随机种子影响复现。"""
    h = 2166136261
    for ch in token:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return h


def _bucket_add(vec: np.ndarray, token: str, value: float = 1.0):
    """把 token 累加到固定维度向量的哈希桶。"""
    idx = _stable_hash(token) % vec.shape[0]
    vec[idx] += value


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """做 L2 归一化，避免不同长度分子尺度差异过大。"""
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def _parse_smiles_tokens_fallback(smiles: str) -> Tuple[List[str], List[str]]:
    """无 RDKit 时的简化解析：抽取原子符号和键符号。"""
    atom_tokens = []
    bond_tokens = []

    # 常见元素优先双字符匹配，避免 Cl 被拆成 C + l。
    i = 0
    while i < len(smiles):
        if i + 1 < len(smiles) and smiles[i:i + 2] in {"Cl", "Br", "Si", "Na", "Ca", "Li", "Mg", "Al"}:
            atom_tokens.append(smiles[i:i + 2])
            i += 2
            continue
        ch = smiles[i]
        if ch.isalpha() and ch.upper() == ch:
            atom_tokens.append(ch)
        if ch in {"-", "=", "#", ":"}:
            bond_tokens.append(ch)
        i += 1
    return atom_tokens, bond_tokens


def encode_atom_bond(smiles: str, dim: int) -> np.ndarray:
    """原子/化学键聚合编码。"""
    vec = np.zeros(dim, dtype=np.float32)
    if not smiles:
        return vec

    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return vec

        for atom in mol.GetAtoms():
            _bucket_add(vec, f"A_{atom.GetSymbol()}", 1.0)
            _bucket_add(vec, f"D_{atom.GetDegree()}", 0.3)
            _bucket_add(vec, f"C_{atom.GetFormalCharge()}", 0.2)
            if atom.GetIsAromatic():
                _bucket_add(vec, "AROMATIC_ATOM", 0.5)

        for bond in mol.GetBonds():
            _bucket_add(vec, f"B_{str(bond.GetBondType())}", 1.0)
            if bond.GetIsConjugated():
                _bucket_add(vec, "CONJ_BOND", 0.3)
            if bond.IsInRing():
                _bucket_add(vec, "RING_BOND", 0.3)
    else:
        atom_tokens, bond_tokens = _parse_smiles_tokens_fallback(smiles)
        for token in atom_tokens:
            _bucket_add(vec, f"A_{token}", 1.0)
        for token in bond_tokens:
            _bucket_add(vec, f"B_{token}", 1.0)

    return _l2_normalize(vec).astype(np.float32)


def encode_fingerprint(smiles: str, dim: int) -> np.ndarray:
    """分子指纹编码，优先 Morgan 指纹。"""
    vec = np.zeros(dim, dtype=np.float32)
    if not smiles:
        return vec

    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return vec
        # 固定 2048 位再投影，减少哈希碰撞
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=_FP_NBITS)
        arr = np.zeros((_FP_NBITS,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return _l2_normalize(_fit_dim(arr, dim))

    # 无 RDKit 时，用字符 n-gram 哈希近似。
    chars = list(smiles)
    for n in (2, 3):
        for i in range(max(0, len(chars) - n + 1)):
            gram = "".join(chars[i:i + n])
            _bucket_add(vec, f"G{n}_{gram}", 1.0)
    return _l2_normalize(vec).astype(np.float32)


def _fit_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """把向量裁剪/补零到目标维度。"""
    if vec.shape[0] == target_dim:
        return vec.astype(np.float32)
    if vec.shape[0] > target_dim:
        return vec[:target_dim].astype(np.float32)
    out = np.zeros(target_dim, dtype=np.float32)
    out[:vec.shape[0]] = vec
    return out


def encode_physchem(smiles: str, dim: int) -> np.ndarray:
    """物化描述符 + MACCS Keys 编码。

    13 个标量描述符经 clip+min-max 归一化后与 MACCS 167 位拼接（共 180 维），
    再用 _fit_dim 截断/补零到 dim，最后 L2 归一化。
    无 RDKit 时返回全零向量。
    """
    vec = np.zeros(dim, dtype=np.float32)
    if not smiles or not HAS_RDKIT:
        return vec

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return vec

    # 计算 Gasteiger 电荷（MaxPartialCharge/MinPartialCharge 依赖此步骤）
    AllChem.ComputeGasteigerCharges(mol)

    # 13 个标量描述符（顺序与 _PHYSCHEM_CLIPS 对应）
    raw_scalars = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        float(mol.GetNumHeavyAtoms()),
        float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
    ]

    # clip + min-max 归一化到 [0, 1]，NaN 补 0
    scalars = np.array(raw_scalars, dtype=np.float64)
    scalars = np.nan_to_num(scalars, nan=0.0)
    for j, (lo, hi) in enumerate(_PHYSCHEM_CLIPS):
        scalars[j] = (np.clip(scalars[j], lo, hi) - lo) / (hi - lo + 1e-12)

    # MACCS Keys 167 位（二值，直接转 float）
    maccs = np.zeros(167, dtype=np.float32)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    Chem.DataStructs.ConvertToNumpyArray(maccs_fp, maccs)

    # 拼接后投影到 dim
    combined = np.concatenate([scalars.astype(np.float32), maccs])
    return _l2_normalize(_fit_dim(combined, dim))


def load_smiles_map(smiles_csv_path: str) -> Dict[str, str]:
    """读取 DrugBank ID -> SMILES 对照表。"""
    path = Path(smiles_csv_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "DrugBank ID" not in df.columns or "SMILES" not in df.columns:
        return {}

    df = df[["DrugBank ID", "SMILES"]].copy()
    df["DrugBank ID"] = df["DrugBank ID"].astype(str).str.strip()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df = df[(df["DrugBank ID"] != "") & (df["SMILES"] != "") & (df["SMILES"] != "nan")]
    df = df.drop_duplicates(subset=["DrugBank ID"], keep="first")

    return {k: v for k, v in zip(df["DrugBank ID"].tolist(), df["SMILES"].tolist())}


def _to_numeric_embedding_list(row: pd.Series, id_col: str, feat_cols: List[str]) -> np.ndarray:
    """把一行 MolFormer 记录转成数值向量。"""
    if feat_cols:
        vals = pd.to_numeric(row[feat_cols], errors="coerce").to_numpy(dtype=np.float32)
        return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    if "embedding" in row.index:
        raw = row["embedding"]
        if isinstance(raw, str):
            try:
                arr = np.array(ast.literal_eval(raw), dtype=np.float32)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass

    vals = []
    for col, v in row.items():
        if col == id_col:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return np.array([], dtype=np.float32)
    arr = np.array(vals, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_molformer_feature_cols(df: pd.DataFrame, id_col: str) -> List[str]:
    """优先识别宽表格式的 mf_* 列。"""
    mf_cols = [col for col in df.columns if col != id_col and str(col).startswith("mf_")]
    if mf_cols:
        return sorted(mf_cols)

    if "embedding" in df.columns:
        return []

    return [
        col for col in df.columns
        if col != id_col and pd.api.types.is_numeric_dtype(df[col])
    ]


def load_molformer_map(molformer_feat_path: str) -> Dict[str, np.ndarray]:
    """读取外部分子预训练向量表。"""
    if not molformer_feat_path:
        return {}
    path = Path(molformer_feat_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if len(df.columns) == 0:
        return {}

    id_col = "DrugBank ID" if "DrugBank ID" in df.columns else df.columns[0]
    feat_cols = _resolve_molformer_feature_cols(df, id_col)
    out = {}
    for _, row in df.iterrows():
        drug_id = str(row[id_col]).strip()
        if not drug_id:
            continue
        vec = _to_numeric_embedding_list(row, id_col, feat_cols)
        if vec.size > 0:
            out[drug_id] = vec
    return out


def encode_smiles(smiles: str, dim: int, encoder_type: str, molformer_vec: np.ndarray = None) -> np.ndarray:
    """统一编码入口。
       1. atom_bond: 原子键聚合编码。
       2. fingerprint: 分子指纹编码。
       3. physchem: 物化描述符编码。
       4. molformer: MolFormer 预训练向量编码。
          - 优先使用外部文件提供的真实维度向量，缺失时回退到 hybrid。
          - 没有 MolFormer 向量时回退到 hybrid，避免整列全零。
       5. hybrid: 原子键聚合 + 分子指纹 + 物化描述符三路均分融合。
    """
    mode = (encoder_type or "hybrid").lower()

    if mode == "atom_bond":
        return encode_atom_bond(smiles, dim)

    if mode == "fingerprint":
        return encode_fingerprint(smiles, dim)

    if mode == "physchem":
        return encode_physchem(smiles, dim)

    if mode == "molformer":
        if molformer_vec is not None and molformer_vec.size > 0:
            arr = np.nan_to_num(molformer_vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if arr.shape[0] == dim:
                return _l2_normalize(arr)
            return _l2_normalize(_fit_dim(arr, dim))
        # 没有 MolFormer 向量时回退到 hybrid，避免整列全零。
        atom_vec = encode_atom_bond(smiles, dim)
        fp_vec = encode_fingerprint(smiles, dim)
        pc_vec = encode_physchem(smiles, dim)
        w = 1.0 / 3.0
        return _l2_normalize(w * atom_vec + w * fp_vec + w * pc_vec).astype(np.float32)

    # hybrid: 原子键聚合 + 分子指纹 + 物化描述符三路均分融合。
    atom_vec = encode_atom_bond(smiles, dim)
    fp_vec = encode_fingerprint(smiles, dim)
    pc_vec = encode_physchem(smiles, dim)
    w = 1.0 / 3.0
    return _l2_normalize(w * atom_vec + w * fp_vec + w * pc_vec).astype(np.float32)


def build_drug_struct_matrix(
    ordered_drug_ids: List[str],
    smiles_csv_path: str,
    encoder_type: str = "hybrid",
    struct_dim: int = 128,
    molformer_feat_path: str = "",
) -> Tuple[np.ndarray, Dict[str, int]]:
    """按药物顺序构建结构特征矩阵。"""
    struct_dim = int(struct_dim)
    if struct_dim <= 0:
        raise ValueError("drug_struct_dim 必须 > 0")

    smiles_map = load_smiles_map(smiles_csv_path)
    use_molformer = encoder_type.lower() == "molformer"
    molformer_map = load_molformer_map(molformer_feat_path) if use_molformer else {}
    loaded_molformer_dim = 0
    if molformer_map:
        first_vec = next(iter(molformer_map.values()))
        loaded_molformer_dim = int(first_vec.shape[0])

    # MolFormer 模式优先使用外部文件的真实维度。
    real_dim = loaded_molformer_dim if loaded_molformer_dim > 0 else struct_dim
    mat = np.zeros((len(ordered_drug_ids), real_dim), dtype=np.float32)
    hit_smiles = 0
    hit_molformer = 0
    missing_molformer_ids = []

    for i, drug_id in enumerate(ordered_drug_ids):
        smiles = smiles_map.get(drug_id, "")
        if smiles:
            hit_smiles += 1
        molformer_vec = molformer_map.get(drug_id)
        if molformer_vec is not None and molformer_vec.size > 0:
            hit_molformer += 1
        elif use_molformer and loaded_molformer_dim > 0:
            missing_molformer_ids.append(drug_id)
        mat[i] = encode_smiles(smiles, real_dim, encoder_type, molformer_vec=molformer_vec)

    stat = {
        "total_drug": len(ordered_drug_ids),
        "hit_smiles": hit_smiles,
        "hit_molformer": hit_molformer,
        "rdkit_enabled": int(HAS_RDKIT),
        "loaded_molformer_dim": loaded_molformer_dim,
        "missing_molformer_ids": missing_molformer_ids,
    }
    return mat, stat

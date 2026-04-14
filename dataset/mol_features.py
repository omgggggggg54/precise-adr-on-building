import ast
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


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
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=dim)
        arr = np.zeros((dim,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)

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


def _to_numeric_embedding_list(row: pd.Series, id_col: str) -> np.ndarray:
    """把一行 MolFormer 记录转成数值向量。"""
    if "embedding" in row.index:
        raw = row["embedding"]
        if isinstance(raw, str):
            try:
                arr = np.array(ast.literal_eval(raw), dtype=np.float32)
                return arr
            except Exception:
                pass

    vals = []
    for col, v in row.items():
        if col == id_col:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not math.isnan(float(v)):
            vals.append(float(v))
    return np.array(vals, dtype=np.float32)


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
    out = {}
    for _, row in df.iterrows():
        drug_id = str(row[id_col]).strip()
        if not drug_id:
            continue
        vec = _to_numeric_embedding_list(row, id_col)
        if vec.size > 0:
            out[drug_id] = vec
    return out


def encode_smiles(smiles: str, dim: int, encoder_type: str, molformer_vec: np.ndarray = None) -> np.ndarray:
    """统一编码入口。"""
    mode = (encoder_type or "hybrid").lower()

    if mode == "atom_bond":
        return encode_atom_bond(smiles, dim)

    if mode == "fingerprint":
        return encode_fingerprint(smiles, dim)

    if mode == "molformer":
        if molformer_vec is not None and molformer_vec.size > 0:
            return _l2_normalize(_fit_dim(molformer_vec, dim))
        # 没有 MolFormer 向量时回退到 hybrid，避免整列全零。
        atom_vec = encode_atom_bond(smiles, dim)
        fp_vec = encode_fingerprint(smiles, dim)
        return _l2_normalize(0.5 * atom_vec + 0.5 * fp_vec).astype(np.float32)

    # hybrid: 原子键聚合 + 分子指纹融合。
    atom_vec = encode_atom_bond(smiles, dim)
    fp_vec = encode_fingerprint(smiles, dim)
    return _l2_normalize(0.5 * atom_vec + 0.5 * fp_vec).astype(np.float32)


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
    molformer_map = load_molformer_map(molformer_feat_path) if encoder_type.lower() == "molformer" else {}

    mat = np.zeros((len(ordered_drug_ids), struct_dim), dtype=np.float32)
    hit_smiles = 0
    hit_molformer = 0

    for i, drug_id in enumerate(ordered_drug_ids):
        smiles = smiles_map.get(drug_id, "")
        if smiles:
            hit_smiles += 1
        molformer_vec = molformer_map.get(drug_id)
        if molformer_vec is not None and molformer_vec.size > 0:
            hit_molformer += 1
        mat[i] = encode_smiles(smiles, struct_dim, encoder_type, molformer_vec=molformer_vec)

    stat = {
        "total_drug": len(ordered_drug_ids),
        "hit_smiles": hit_smiles,
        "hit_molformer": hit_molformer,
        "rdkit_enabled": int(HAS_RDKIT),
    }
    return mat, stat

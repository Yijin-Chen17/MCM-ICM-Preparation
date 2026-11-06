"""
MCM/ICM Preprocessing Series: 02 - Strings & Categorical Toolkit
美国大学生数学建模竞赛数据预处理工具箱：02 - 字符串清洗与类别型特征手术刀

设计原则：
- 高度工程化与抗造性（Defensive Programming）：
  * 自动剥离数字外围的乱七八糟字符（$, %, 逗号千分位、单位、会计负数括号）
  * 独热编码（One-Hot）内置罕见长尾合并（min_freq 机制），杜绝 100 列稀疏矩阵导致的维度灾难
  * 目标编码（Target Encoding）纯手打内置 5 折交叉验证平滑，绝无数据泄露
  * 一键导出美赛论文专用的“变量编码字典 (LaTeX Codebook)”
"""

import re
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def clean_numeric_string(
    series: pd.Series,
    strip_chars: Optional[List[str]] = None,
    default_val: float = np.nan
) -> pd.Series:
    """
    清洗“披着字符串外衣的假数值列”（如 "$1,234.50", "85.2%", "(120)", "15.3km"）。
    
    参数：
    ----------
    series : pd.Series
        待清洗的文本列
    strip_chars : list of str, 可选
        额外需显式剔除的特定物理单位或字符
    default_val : float, 默认 np.nan
        无法解析成浮点数时的兜底填充值
        
    返回：
    ----------
    pd.Series: float64 类型的纯数值序列
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = series.astype(str).str.strip()
    
    # 将常见的空文本表示转为 NaN
    na_tokens = {'nan', 'none', 'null', 'na', 'n/a', '-', '', 'missing'}
    is_na = s.str.lower().isin(na_tokens)
    
    # 额外剔除用户自定义单位
    if strip_chars:
        for ch in strip_chars:
            s = s.str.replace(ch, '', regex=False)
            
    # 处理会计负数括号，例如 "(123.45)" -> "-123.45"
    s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
    
    # 移除非数字、非小数点、非负号的所有字符（剥离 $, %, 逗号等）
    s = s.str.replace(r'[^\d.-]', '', regex=True)
    
    # 转换为数值类型
    numeric_s = pd.to_numeric(s, errors='coerce')
    numeric_s[is_na] = default_val
    if pd.isna(default_val):
        numeric_s = numeric_s.fillna(np.nan)
    else:
        numeric_s = numeric_s.fillna(default_val)
        
    return numeric_s


def encode_binary(
    series: pd.Series,
    mapping: Optional[Dict[str, int]] = None,
    default_val: float = np.nan
) -> pd.Series:
    """
    将二元分类字符串（如 "Yes"/"No", "Win"/"Loss", "True"/"False"）极速对齐为 0/1 数值。
    
    若不提供 mapping，则自动探测序列中的唯二非空类别：
    出现频次更高的赋为 1，另一类别赋为 0。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = series.copy()
    
    if mapping is None:
        # 自动推断
        valid_vals = s.dropna().unique().tolist()
        if len(valid_vals) != 2:
            raise ValueError(f"自动二元推断失败：该列包含 {len(valid_vals)} 个唯一值，必须恰好为 2 个才能自动推断。请传入显式 mapping 参数。唯一值: {valid_vals}")
        # 频次高者为 1，次者为 0
        counts = s.value_counts(ascending=False)
        mapping = {counts.index[0]: 1, counts.index[1]: 0}
        
    # 映射处理
    result = s.map(mapping)
    if pd.isna(default_val):
        return result.astype(float)
    return result.fillna(default_val).astype(float)


def encode_ordinal(
    series: pd.Series,
    order_list: List[str],
    default_val: float = np.nan
) -> pd.Series:
    """
    有序类别编码（Ordinal Encoding）：将具有显式自然大小等级的字符串按顺序映射为 0, 1, 2, ...
    
    参数：
    ----------
    order_list : list of str
        严格按从低到高排列的类别名称，如 ['Low', 'Medium', 'High', 'Critical']
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    mapping = {val: idx for idx, val in enumerate(order_list)}
    result = series.map(mapping)
    if pd.isna(default_val):
        return result.astype(float)
    return result.fillna(default_val).astype(float)


def encode_onehot(
    df: pd.DataFrame,
    column: str,
    min_freq: float = 0.03,
    other_label: str = 'Other',
    drop_first: bool = False,
    prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    智能独热编码（One-Hot Encoding）：
    1. 自动对长尾低频类别进行截断并归并为 `other_label`，彻底防范维度灾难；
    2. 支持 `drop_first=True`，彻底杜绝线性基准模型中的多重共线性（Dummy Variable Trap）；
    3. 保留原 DataFrame 索引与非目标列，原目标列替换为独热衍生列。
    
    参数：
    ----------
    df : pd.DataFrame
        数据表
    column : str
        需独热编码的目标列名
    min_freq : float, 默认 0.03 (3%)
        频率阈值。频次占比小于此值的罕见项将被归为 other_label
    other_label : str, 默认 'Other'
        低频项归并名称
    drop_first : bool, 默认 False
        是否丢弃第一个类别基准列（配合回归基准模型使用时应设为 True）
    prefix : str, 可选
        生成列名的前缀，默认即为 column 名
    """
    if column not in df.columns:
        raise KeyError(f"列 '{column}' 不存在于 DataFrame 中")
        
    res_df = df.copy()
    s = res_df[column].astype(str)
    
    # 统计频率
    freqs = s.value_counts(normalize=True)
    rare_categories = freqs[freqs < min_freq].index.tolist()
    
    if rare_categories:
        # 将低频项替换为 Other
        s = s.replace(rare_categories, other_label)
        
    # 执行独热
    col_prefix = prefix if prefix is not None else column
    dummies = pd.get_dummies(s, prefix=col_prefix, drop_first=drop_first, dtype=int)
    
    # 拼接并删除原列
    col_idx = res_df.columns.get_loc(column)
    res_df = res_df.drop(columns=[column])
    
    for i, dummy_col in enumerate(dummies.columns):
        res_df.insert(col_idx + i, dummy_col, dummies[dummy_col])
        
    return res_df


def encode_frequency(
    series: pd.Series,
    normalize: bool = True
) -> pd.Series:
    """
    频数编码（Frequency / Count Encoding）：用该类别在整体样本中出现的频率或绝对次数替代类别字符串。
    适合树模型或高基数特征。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    freq_map = series.value_counts(normalize=normalize).to_dict()
    return series.map(freq_map).astype(float)


def encode_target_cv(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42
) -> pd.Series:
    """
    平滑目标编码（Target Encoding，Out-of-Fold 交叉验证版）：
    利用 K 折交叉验证计算每个类别的目标变量平滑均值，彻底防止特征泄露（Data Leakage）。
    
    公式：
    S_c = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)
    """
    if cat_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"列 {cat_col} 或 {target_col} 不存在于 DataFrame 中")
        
    df_clean = df[[cat_col, target_col]].copy()
    global_mean = float(df_clean[target_col].mean())
    n_samples = len(df_clean)
    
    # 打乱索引生成交叉验证 Fold
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    folds = np.array_split(indices, n_splits)
    
    encoded_vals = np.zeros(n_samples, dtype=float)
    
    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        train_idx = np.setdiff1d(indices, val_idx)
        
        train_df = df_clean.iloc[train_idx]
        val_df = df_clean.iloc[val_idx]
        
        # 在训练折内计算统计量
        stats_by_cat = train_df.groupby(cat_col)[target_col].agg(['count', 'mean'])
        
        # 平滑公式
        stats_by_cat['smooth_val'] = (
            stats_by_cat['count'] * stats_by_cat['mean'] + smoothing * global_mean
        ) / (stats_by_cat['count'] + smoothing)
        
        smooth_dict = stats_by_cat['smooth_val'].to_dict()
        
        # 映射到验证折，未出现过的类别使用 global_mean 兜底
        val_encoded = val_df[cat_col].map(smooth_dict).fillna(global_mean).to_numpy()
        encoded_vals[val_idx] = val_encoded
        
    return pd.Series(encoded_vals, index=df.index, name=f"{cat_col}_target_enc")


def export_codebook_latex(
    mapping_dict: Dict[str, Dict[str, Union[int, float]]],
    caption: str = "Categorical Variable Numerical Encoding Codebook",
    label: str = "tab:codebook"
) -> str:
    """
    自动导出符合美赛论文规范的“变量数值编码映射对照三线表 (LaTeX)”。
    
    示例输入 mapping_dict:
    {
        "Surface_Type": {"Grass": 0, "Clay": 1, "Hard": 2},
        "Match_Result": {"Loss": 0, "Win": 1}
    }
    """
    rows = []
    for var_name, mapping in mapping_dict.items():
        pairs = [f"{k} $\\to$ {v}" for k, v in mapping.items()]
        pairs_str = ", ".join(pairs)
        # 转义变量名中的下划线
        var_escaped = str(var_name).replace("_", r"\_")
        rows.append(f"  \\textbf{{{var_escaped}}} & {len(mapping)} & {pairs_str} \\\\")
        
    body = "\n".join(rows)
    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lcl}}
\\toprule
\\textbf{{Variable Name}} & \\textbf{{Levels}} & \\textbf{{Numerical Mapping Definition}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    return latex_code

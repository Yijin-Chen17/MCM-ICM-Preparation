"""
MCM/ICM Preprocessing Series: 03 - Missing & Outlier Toolkit
美国大学生数学建模竞赛数据预处理工具箱：03 - 缺失值与异常值精细治理

设计原则：
- 高度工程化与抗造性（Defensive Programming）：
  * 异常检测与治理彻底解耦：先返回布尔掩码（Mask）供人工复核，再由人决定是缩尾截断还是剔除
  * 重点推崇缩尾处理（Winsorization）：在保留全部样本量的前提下抑制极端值影响
  * 插值提供 PCHIP（单调保形插值），彻底解决传统样条在物理模型中的反常识过冲震荡
  * 容忍零方差、全空列、端点缺失（自动边界平滑兜底）
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import PchipInterpolator, make_interp_spline


def detect_outliers_iqr(
    series: pd.Series,
    k: float = 1.5
) -> pd.Series:
    """
    基于 Tukey 四分位距（IQR）法则检测异常值，返回布尔掩码。
    
    参数：
    ----------
    series : pd.Series
        待检测数值序列
    k : float, 默认 1.5
        离群倍数（1.5 为轻度异常，3.0 为极端异常）
        
    返回：
    ----------
    pd.Series (bool): True 表示为异常离群点，False/NaN 保持为正常/缺失
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce')
    valid = s.dropna()
    
    if len(valid) < 4:
        return pd.Series(False, index=series.index)
        
    q25 = float(valid.quantile(0.25))
    q75 = float(valid.quantile(0.75))
    iqr = q75 - q25
    
    if iqr < 1e-12:
        # IQR 几乎为 0，退化为无异常
        return pd.Series(False, index=series.index)
        
    lower_bound = q25 - k * iqr
    upper_bound = q75 + k * iqr
    
    mask = (s < lower_bound) | (s > upper_bound)
    return mask.fillna(False)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """
    基于 3-Sigma / Z-score 准则检测异常值，返回布尔掩码。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce')
    valid = s.dropna()
    
    if len(valid) < 3:
        return pd.Series(False, index=series.index)
        
    mean_val = float(valid.mean())
    std_val = float(valid.std(ddof=1))
    
    if std_val < 1e-12:
        return pd.Series(False, index=series.index)
        
    z_scores = np.abs((s - mean_val) / std_val)
    return (z_scores > threshold).fillna(False)


def winsorize_iqr(
    series: pd.Series,
    k: float = 1.5
) -> pd.Series:
    """
    Tukey 四分位缩尾截断处理（Winsorization）：
    将超出 [Q1 - k*IQR, Q3 + k*IQR] 的极端值拉回至边界上，既消除了离群值破坏，又彻底保全了宝贵的样本量。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce').copy()
    valid = s.dropna()
    
    if len(valid) < 4:
        return s
        
    q25 = float(valid.quantile(0.25))
    q75 = float(valid.quantile(0.75))
    iqr = q75 - q25
    
    if iqr < 1e-12:
        return s
        
    lower_bound = q25 - k * iqr
    upper_bound = q75 + k * iqr
    
    return s.clip(lower=lower_bound, upper=upper_bound)


def winsorize_quantiles(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99
) -> pd.Series:
    """
    分位数双端缩尾处理（Quantile Winsorization）：
    将小于 lower 分位数或大于 upper 分位数的值拉回至分位数边界（常用 [1%, 99%] 或 [2.5%, 97.5%]）。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce').copy()
    valid = s.dropna()
    
    if len(valid) < 4:
        return s
        
    low_val = float(valid.quantile(lower))
    high_val = float(valid.quantile(upper))
    
    return s.clip(lower=low_val, upper=high_val)


def impute_statistical(
    series: pd.Series,
    strategy: str = 'median',
    fill_value: Optional[Union[float, int, str]] = None
) -> pd.Series:
    """
    基础统计值缺失填补。
    
    参数：
    ----------
    strategy : str
        可选 'median'（中位数，抗偏态默认推荐）、'mean'（均值）、'mode'（众数）、
        'ffill'（前向填充，时序推荐）、'bfill'（后向填充）、'constant'（常数填充）
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = series.copy()
    if strategy == 'median':
        val = s.median()
        return s.fillna(val)
    elif strategy == 'mean':
        val = s.mean()
        return s.fillna(val)
    elif strategy == 'mode':
        mode_vals = s.mode()
        val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
        return s.fillna(val)
    elif strategy == 'ffill':
        return s.ffill().bfill()
    elif strategy == 'bfill':
        return s.bfill().ffill()
    elif strategy == 'constant':
        return s.fillna(fill_value if fill_value is not None else 0)
    else:
        raise ValueError(f"不支持的 strategy: {strategy}，可选: 'median', 'mean', 'mode', 'ffill', 'bfill', 'constant'")


def interpolate_series(
    series: pd.Series,
    method: str = 'pchip'
) -> pd.Series:
    """
    高保真数学插值填补（专为时序、物理、仿真轨迹连续补齐设计）。
    
    参数：
    ----------
    method : str
        可选：
        - 'pchip': 分段三次埃尔米特保形插值（保持局部单调性，绝不出现虚假极值或负值过冲，物理建模首选！）
        - 'linear': 线性插值
        - 'spline': 三次样条插值
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce').copy()
    
    # 提取已知点坐标与未知点坐标
    valid_mask = s.notna().to_numpy()
    if valid_mask.sum() < 2:
        return s
        
    x_all = np.arange(len(s))
    x_valid = x_all[valid_mask]
    y_valid = s.to_numpy()[valid_mask]
    
    if method == 'linear':
        return s.interpolate(method='linear').ffill().bfill()
        
    elif method == 'pchip':
        # PCHIP 要求已知点至少 2 个且严格递增
        if len(x_valid) < 2:
            return s.ffill().bfill()
        try:
            pchip = PchipInterpolator(x_valid, y_valid, extrapolate=True)
            y_interp = pchip(x_all)
            res = pd.Series(y_interp, index=series.index)
            # 边界若有外推极值，以原数据边界安全截断
            return res.clip(lower=y_valid.min(), upper=y_valid.max())
        except Exception:
            return s.interpolate(method='linear').ffill().bfill()
            
    elif method == 'spline':
        if len(x_valid) < 4:
            return s.interpolate(method='linear').ffill().bfill()
        try:
            spline = make_interp_spline(x_valid, y_valid, k=3)
            y_interp = spline(x_all)
            return pd.Series(y_interp, index=series.index)
        except Exception:
            return s.interpolate(method='linear').ffill().bfill()
    else:
        raise ValueError(f"不支持的插值方法: {method}，可选: 'pchip', 'linear', 'spline'")


def impute_knn(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    n_neighbors: int = 5
) -> pd.Series:
    """
    纯 NumPy / SciPy 实现的 KNN 多维相似度插补器（免依赖、高稳健性）。
    通过在完整特征空间计算距离，找出与缺失样本最相似的 K 个邻居的加权均值进行填补。
    """
    if target_col not in df.columns:
        raise KeyError(f"列 {target_col} 不存在")
        
    valid_features = [c for c in feature_cols if c in df.columns and c != target_col]
    if not valid_features:
        raise ValueError("必须提供至少一个用于计算距离的辅助特征列")
        
    res_series = df[target_col].copy()
    missing_mask = res_series.isna()
    
    if not missing_mask.any():
        return res_series
        
    # 提取特征矩阵并做标准化
    X = df[valid_features].copy()
    X = X.fillna(X.median()) # 辅助特征若有轻微缺失以中位数兜底
    
    x_mean = X.mean()
    x_std = X.std(ddof=1).replace(0, 1) # 防除以0
    X_norm = ((X - x_mean) / x_std).to_numpy()
    
    known_idx = np.where(~missing_mask)[0]
    miss_idx = np.where(missing_mask)[0]
    
    if len(known_idx) == 0:
        return res_series.fillna(0)
        
    k = min(n_neighbors, len(known_idx))
    y_known = res_series.iloc[known_idx].to_numpy()
    
    # 逐行填补
    for idx in miss_idx:
        q_vec = X_norm[idx, :]
        dists = np.linalg.norm(X_norm[known_idx] - q_vec, axis=1)
        k_nearest = np.argpartition(dists, k - 1)[:k]
        
        nearest_dists = dists[k_nearest]
        nearest_y = y_known[k_nearest]
        
        # 距离倒数加权
        weights = 1.0 / (nearest_dists + 1e-6)
        imputed_val = np.sum(weights * nearest_y) / np.sum(weights)
        res_series.iloc[idx] = float(imputed_val)
        
    return res_series


def plot_outlier_inspection(
    series_raw: pd.Series,
    series_clean: Optional[pd.Series] = None,
    title: str = "Outlier Inspection & Winsorization Effect",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    绘制美赛学术级异常治理前后箱线图与核密度分布对比图。
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300 if save_path else 100)
    
    raw_valid = series_raw.dropna().to_numpy()
    data_to_plot = [raw_valid]
    labels = ["Raw"]
    
    if series_clean is not None:
        clean_valid = series_clean.dropna().to_numpy()
        data_to_plot.append(clean_valid)
        labels.append("Cleaned (Winsorized)")
        
    # 1. 箱线图
    bp = axes[0].boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                         boxprops=dict(facecolor="#BDD7EE", color="#1F4E79"),
                         medianprops=dict(color="#C00000", linewidth=1.5),
                         flierprops=dict(marker='o', markersize=3, markerfacecolor='red', alpha=0.5))
    axes[0].set_title("Boxplot Comparison", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Value Scale", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. KDE 密度分布图
    for data, lbl, col in zip(data_to_plot, labels, ["#E06666", "#2B7A78"]):
        if len(data) > 1 and np.std(data) > 1e-6:
            density = stats.gaussian_kde(data)
            xs = np.linspace(min(data), max(data), 200)
            axes[1].plot(xs, density(xs), label=lbl, color=col, linewidth=1.8)
            axes[1].fill_between(xs, density(xs), alpha=0.2, color=col)
            
    axes[1].set_title("Density Distribution (KDE)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Value", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

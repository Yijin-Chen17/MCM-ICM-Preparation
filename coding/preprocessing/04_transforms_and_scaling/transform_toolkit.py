"""
MCM/ICM Preprocessing Series: 04 - Transforms & Scaling Toolkit
美国大学生数学建模竞赛数据预处理工具箱：04 - 分布正态化与综合评价无量纲化

设计原则：
- 高度工程化与抗造性（Defensive Programming）：
  * Box-Cox 自动探测零值与负数，支持安全正向平移，绝不因 math domain error 崩溃
  * 评价无量纲化覆盖数学建模最经典的四大类型：极大型（效益）、极小型（成本）、区间最优型、TOPSIS 专用向量归一化
  * 针对熵权法自动引入微小偏移量 eps，彻底规避计算信息熵时出现 ln(0) 报错
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def transform_boxcox(
    series: pd.Series,
    lmbda: Optional[float] = None,
    shift_if_needed: bool = True
) -> Tuple[pd.Series, float, float]:
    """
    Box-Cox 幂变换（将右偏、长尾分布自适应恢复为近似正态分布）。
    
    返回：
    ----------
    (pd.Series, float, float): (变换后的序列, 最优参数 lambda, 平移常数 c)
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce').copy()
    valid_mask = s.notna()
    y = s[valid_mask].to_numpy(dtype=float)
    
    if len(y) < 3:
        return s, 1.0, 0.0
        
    min_val = float(np.min(y))
    shift_c = 0.0
    
    if min_val <= 0:
        if not shift_if_needed:
            raise ValueError(f"Box-Cox 要求所有数据严格大于 0，检测到最小值 {min_val}。请开启 shift_if_needed=True。")
        shift_c = abs(min_val) + 1.0
        y = y + shift_c
        
    if lmbda is None:
        y_trans, opt_lambda = stats.boxcox(y)
    else:
        y_trans = stats.boxcox(y, lmbda=lmbda)
        opt_lambda = lmbda
        
    res = pd.Series(index=series.index, dtype=float)
    res[valid_mask] = y_trans
    return res, float(opt_lambda), float(shift_c)


def transform_yeojohnson(
    series: pd.Series,
    lmbda: Optional[float] = None
) -> Tuple[pd.Series, float]:
    """
    Yeo-Johnson 幂变换（无需平移，原生支持零与负数的自适应正态化变换）。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce').copy()
    valid_mask = s.notna()
    y = s[valid_mask].to_numpy(dtype=float)
    
    if len(y) < 3:
        return s, 1.0
        
    if lmbda is None:
        y_trans, opt_lambda = stats.yeojohnson(y)
    else:
        y_trans = stats.yeojohnson(y, lmbda=lmbda)
        opt_lambda = lmbda
        
    res = pd.Series(index=series.index, dtype=float)
    res[valid_mask] = y_trans
    return res, float(opt_lambda)


def scale_benefit(series: pd.Series) -> pd.Series:
    """
    效益型（极大型）指标正向化映射：指标值越大越好，线性归一化至 [0, 1]。
    公式：x* = (x - min) / (max - min)
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors='coerce').copy()
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or max_val - min_val < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (s - min_val) / (max_val - min_val)


def scale_cost(series: pd.Series, method: str = 'difference') -> pd.Series:
    """
    成本型（极小型）指标正向化映射：指标值越小越好，逆转并归一化至 [0, 1]。
    
    参数：
    ----------
    method : str
        - 'difference': 差值逆转 x* = (max - x) / (max - min) (推荐，保留线性间距)
        - 'reciprocal': 倒数法 x* = 1 / x 并归一化 (适用于速率、排量等物理量)
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors='coerce').copy()
    min_val = s.min()
    max_val = s.max()
    
    if pd.isna(min_val) or max_val - min_val < 1e-12:
        return pd.Series(0.0, index=series.index)
        
    if method == 'difference':
        return (max_val - s) / (max_val - min_val)
    elif method == 'reciprocal':
        # 倒数法前置检查
        if (s <= 0).any():
            # 若有非正数，自动平移后求倒数
            s = s - s.min() + 1.0
        recip = 1.0 / s
        r_min, r_max = recip.min(), recip.max()
        return (recip - r_min) / (r_max - r_min + 1e-12)
    else:
        raise ValueError(f"未知 method: {method}，可选 'difference', 'reciprocal'")


def scale_interval(
    series: pd.Series,
    optimal_range: Tuple[float, float],
    tolerance_range: Optional[Tuple[float, float]] = None
) -> pd.Series:
    """
    区间最优型（居中型）指标正向化映射（如水温 20-25 度最适宜，过高过低均扣分）。
    
    参数：
    ----------
    optimal_range : (q1, q2)
        最佳满分区间，处于 [q1, q2] 内得分恒为 1.0
    tolerance_range : (low_tol, high_tol), 可选
        容忍区间边界。若不提供，自动默认为 (min(x), max(x))
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors='coerce').copy()
    q1, q2 = optimal_range
    if q1 > q2:
        q1, q2 = q2, q1
        
    low_tol = tolerance_range[0] if tolerance_range else float(s.min())
    high_tol = tolerance_range[1] if tolerance_range else float(s.max())
    
    res = pd.Series(0.0, index=series.index, dtype=float)
    
    # 1. 处于最优区间内，直接给满分 1
    in_optimal = (s >= q1) & (s <= q2)
    res[in_optimal] = 1.0
    
    # 2. 低于最优下限 q1，线性扣分
    below_opt = s < q1
    denom_left = q1 - low_tol if q1 - low_tol > 1e-12 else 1.0
    res[below_opt] = 1.0 - (q1 - s[below_opt]) / denom_left
    
    # 3. 高于最优上限 q2，线性扣分
    above_opt = s > q2
    denom_right = high_tol - q2 if high_tol - q2 > 1e-12 else 1.0
    res[above_opt] = 1.0 - (s[above_opt] - q2) / denom_right
    
    # 截断在 [0, 1] 之间
    return res.clip(lower=0.0, upper=1.0)


def normalize_topsis_vector(series: pd.Series) -> pd.Series:
    """
    TOPSIS 专用向量规范化（Vector Normalization）：
    公式：z_i = x_i / sqrt(sum(x_j^2))
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors='coerce').copy()
    norm = np.sqrt(np.sum(s.dropna() ** 2))
    if norm < 1e-12:
        return pd.Series(0.0, index=series.index)
    return s / norm


def normalize_minmax_epsilon(series: pd.Series, eps: float = 1e-4) -> pd.Series:
    """
    熵权法（Entropy Weight Method）专用平移归一化：
    严格映射至 [eps, 1.0]，彻底规避在计算 p * ln(p) 时因 p=0 发生数学未定义的错误。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors='coerce').copy()
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or max_val - min_val < 1e-12:
        return pd.Series(eps, index=series.index)
    scaled = (s - min_val) / (max_val - min_val)
    return scaled * (1.0 - eps) + eps


def plot_distribution_improvement(
    series_before: pd.Series,
    series_after: pd.Series,
    names: Tuple[str, str] = ("Raw (Skewed)", "Transformed (Normalized)"),
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    绘制美赛学术级变换前后偏态改善对比图（KDE直方核密度 + 正态 Q-Q 图）。
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300 if save_path else 100)
    
    raw = series_before.dropna().to_numpy()
    trans = series_after.dropna().to_numpy()
    
    # 1. 核密度直方对比
    axes[0].hist(raw, bins=25, density=True, alpha=0.4, color="#C00000", label=names[0])
    axes[0].hist(trans, bins=25, density=True, alpha=0.4, color="#2B7A78", label=names[1])
    
    if len(raw) > 1 and np.std(raw) > 1e-6:
        kd1 = stats.gaussian_kde(raw)
        xs1 = np.linspace(raw.min(), raw.max(), 100)
        axes[0].plot(xs1, kd1(xs1), color="#C00000", linewidth=1.5)
        
    if len(trans) > 1 and np.std(trans) > 1e-6:
        kd2 = stats.gaussian_kde(trans)
        xs2 = np.linspace(trans.min(), trans.max(), 100)
        axes[0].plot(xs2, kd2(xs2), color="#2B7A78", linewidth=1.5)
        
    axes[0].set_title("Distribution Histogram & KDE", fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Q-Q 正态诊断图
    stats.probplot(trans, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('#2B7A78')
    axes[1].get_lines()[0].set_markeredgecolor('#2B7A78')
    axes[1].get_lines()[0].set_markersize(4.0)
    axes[1].get_lines()[1].set_color('#C00000')
    axes[1].get_lines()[1].set_linewidth(1.8)
    axes[1].set_title("Transformed Normal Q-Q Plot", fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

"""
MCM/ICM Preprocessing Series: 05 - Timeseries & Momentum Toolkit
美国大学生数学建模竞赛数据预处理工具箱：05 - 动态时序演化与“势头”特征工程 (C 题王牌)

设计原则：
- 高度工程化与抗造性（Defensive Programming）：
  * 专为微观事件流（如 2024 C 题逐球记录、逐日交易、设备监测）设计
  * 彻底杜绝时间穿越（Lookahead Bias）：所有滑动窗口与滞后算子均严格仅利用历史信息
  * 指数记忆衰减算子（EWMA）提供物理/心理学解释，量化“势头（Momentum）”的动态演进
  * 纯向量化实现连续状态（Streak / Run-Length）计算，摆脱低效的 Python for 循环
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_rolling_features(
    series: pd.Series,
    windows: List[int] = [3, 5, 10],
    metrics: List[str] = ['mean', 'std', 'skew', 'max', 'min'],
    min_periods: int = 1
) -> pd.DataFrame:
    """
    多尺度滑动窗口特征生成器（严格无未来函数）。
    
    参数：
    ----------
    series : pd.Series
        时间序列数据
    windows : list of int
        滑动窗口大小列表（如 [3, 5, 10] 分别捕捉超短期、中期与长期的局势状态）
    metrics : list of str
        所需计算的统计指标，可选：'mean', 'std', 'skew', 'max', 'min', 'sum', 'quantile_25', 'quantile_75'
    min_periods : int, 默认 1
        窗口内最小非空观测数（设为 1 可防止序列开头出现大量无意义的 NaN）
        
    返回：
    ----------
    pd.DataFrame: 包含各窗口衍生指标的特征表
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    col_name = str(series.name) if series.name else "val"
    res_df = pd.DataFrame(index=series.index)
    
    for w in windows:
        roller = series.rolling(window=w, min_periods=min_periods)
        for m in metrics:
            feat_name = f"{col_name}_roll_{w}_{m}"
            if m == 'mean':
                res_df[feat_name] = roller.mean()
            elif m == 'std':
                res_df[feat_name] = roller.std().fillna(0.0)
            elif m == 'skew':
                res_df[feat_name] = roller.skew().fillna(0.0)
            elif m == 'max':
                res_df[feat_name] = roller.max()
            elif m == 'min':
                res_df[feat_name] = roller.min()
            elif m == 'sum':
                res_df[feat_name] = roller.sum()
            elif m == 'quantile_25':
                res_df[feat_name] = roller.quantile(0.25)
            elif m == 'quantile_75':
                res_df[feat_name] = roller.quantile(0.75)
            else:
                raise ValueError(f"不支持的指标: {m}")
                
    return res_df


def calc_ewma_momentum(
    series: pd.Series,
    alpha: Optional[float] = 0.3,
    span: Optional[float] = None,
    halflife: Optional[float] = None
) -> pd.Series:
    """
    指数加权移动平均（EWMA）势头算子：
    数学公式：S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    通过设定记忆衰减权重，精准量化比赛中“最近得分/失误对即时心理势头的强烈冲击”。
    
    参数：
    ----------
    series : pd.Series
        输入表现序列（如得分设为 1，失分设为 -1，或即时得失分差）
    alpha : float, 可选
        平滑/记忆更新因子 (0 < alpha <= 1)。alpha 越大说明对最近状态越敏感；默认 0.3
    span : float, 可选
        以跨度表示 (alpha = 2 / (span + 1))
    halflife : float, 可选
        半衰期（经过几个时间步后历史记忆衰减为原先的一半）
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    col_name = str(series.name) if series.name else "momentum"
    
    if halflife is not None:
        ewm_series = series.ewm(halflife=halflife, adjust=False).mean()
    elif span is not None:
        ewm_series = series.ewm(span=span, adjust=False).mean()
    else:
        ewm_series = series.ewm(alpha=alpha, adjust=False).mean()
        
    ewm_series.name = f"{col_name}_ewma"
    return ewm_series


def calc_streak_length(
    series: pd.Series,
    target_value: Union[int, float, str] = 1
) -> pd.Series:
    """
    纯向量化连击/连胜长度统计器（Run-Length Streak Counter）：
    计算当前时间点为止，连续处于 target_value（如连续赢球、连续上涨）的累积次数。
    遇到非 target_value 时自动瞬时归零。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    is_target = (series == target_value).to_numpy(dtype=int)
    streaks = np.zeros(len(is_target), dtype=int)
    
    curr = 0
    for i, val in enumerate(is_target):
        if val == 1:
            curr += 1
        else:
            curr = 0
        streaks[i] = curr
        
    res = pd.Series(streaks, index=series.index, name=f"{series.name}_streak_{target_value}")
    return res


def calc_lag_diff(
    series: pd.Series,
    lags: List[int] = [1, 2],
    diffs: List[int] = [1, 2]
) -> pd.DataFrame:
    """
    滞后项（Lag）与各阶差分（Difference）生成器：
    - Lag 1, Lag 2: 捕获历史基态
    - Diff 1: 变化率/速度（Velocity）
    - Diff 2: 变化加速度（Acceleration）
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    col_name = str(series.name) if series.name else "val"
    res_df = pd.DataFrame(index=series.index)
    
    # 滞后特征
    for lag in lags:
        res_df[f"{col_name}_lag_{lag}"] = series.shift(lag).bfill()
        
    # 差分特征
    for d in diffs:
        res_df[f"{col_name}_diff_{d}"] = series.diff(d).bfill()
        
    return res_df


def resample_regular_series(
    df: pd.DataFrame,
    time_col: str,
    freq: str = '1D',
    agg_dict: Optional[Dict[str, str]] = None,
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    将不规则事件流规整化为等间隔时间序列（如按日、按小时汇总）。
    """
    if time_col not in df.columns:
        raise KeyError(f"时间列 {time_col} 不存在")
        
    temp_df = df.copy()
    temp_df[time_col] = pd.to_datetime(temp_df[time_col])
    temp_df = temp_df.set_index(time_col).sort_index()
    
    if agg_dict is None:
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
        agg_dict = {col: 'mean' for col in numeric_cols}
        
    resampled = temp_df.resample(freq).agg(agg_dict)
    
    if fill_method == 'ffill':
        resampled = resampled.ffill().bfill()
    elif fill_method == 'zero':
        resampled = resampled.fillna(0)
        
    return resampled.reset_index()


def plot_momentum_curve(
    time_series: pd.Series,
    momentum_series: pd.Series,
    event_markers: Optional[pd.Series] = None,
    title: str = "Dynamic Momentum Trajectory & Event Impact",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    绘制美赛出版级学术势头曲线图（双轴叠加大事件标记）。
    """
    fig, ax1 = plt.subplots(figsize=figsize, dpi=300 if save_path else 100)
    
    xs = np.arange(len(momentum_series))
    
    # 绘制势头演化曲线
    line1 = ax1.plot(xs, momentum_series, color="#2B7A78", linewidth=2.0, label="Dynamic Momentum Score")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.6, linewidth=1.0)
    ax1.fill_between(xs, momentum_series, 0, where=(momentum_series >= 0), color="#2B7A78", alpha=0.15)
    ax1.fill_between(xs, momentum_series, 0, where=(momentum_series < 0), color="#C00000", alpha=0.15)
    
    ax1.set_xlabel("Event Sequence / Time Step", fontsize=10)
    ax1.set_ylabel("Momentum Magnitude", fontsize=10, color="#2B7A78")
    ax1.tick_params(axis='y', labelcolor="#2B7A78")
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # 若提供底层大事件（如比赛破发点或关键转折），在双轴标记
    if event_markers is not None:
        ax2 = ax1.twinx()
        line2 = ax2.scatter(xs, event_markers, color="#C00000", marker="^", s=25, alpha=0.7, label="Critical Turning Event")
        ax2.set_ylabel("Event Occurrence", fontsize=10, color="#C00000")
        ax2.tick_params(axis='y', labelcolor="#C00000")
        ax2.set_yticks([0, 1])
        
    ax1.set_title(title, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

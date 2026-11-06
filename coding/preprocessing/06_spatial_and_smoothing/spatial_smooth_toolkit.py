"""
MCM/ICM Preprocessing Series: 06 - Spatial & Smoothing Toolkit
美国大学生数学建模竞赛数据预处理工具箱：06 - 空间几何网格化与信号滤波平滑 (B 题王牌)

设计原则：
- 高度工程化与抗造性（Defensive Programming）：
  * 专为物理仿真、搜寻救援、轨迹规划（如 2024 B 题潜艇失联、2022 B 题无人机飞行）设计
  * Savitzky-Golay 滤波器内置窗口奇数化自动矫正、短序列窗口自适应缩放，绝不抛出参数异常
  * S-G 滤波支持直接解析求导（deriv=1 速度，deriv=2 加速度），彻底杜绝差分噪声爆炸
  * 空间距离计算（Haversine）内置浮点反余弦截断保护，杜绝因极近点精度误差产生 NaN
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def smooth_savitzky_golay(
    series: pd.Series,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
    delta: float = 1.0
) -> pd.Series:
    """
    Savitzky-Golay 局部多项式保形滤波器（物理平滑黄金工具）：
    既能压制传感器高频随机白噪声，又能精准保留波峰、波谷与物理拐点。
    
    参数：
    ----------
    series : pd.Series
        待平滑的物理测量序列（如深度、速度、位置轨迹）
    window_length : int, 默认 11
        滤波滑动窗口大小（必须为正奇数。若传入偶数，算法自动加 1 修正）
    polyorder : int, 默认 2
        局部拟合多项式阶数（必须严格小于窗口长度）
    deriv : int, 默认 0
        直接解析求导阶数：
        - 0: 输出平滑后的物理量（如平滑后的位移 x）
        - 1: 输出平滑后的一阶导数（速度 v = dx/dt）
        - 2: 输出平滑后的二阶导数（加速度 a = d^2x/dt^2）
    delta : float, 默认 1.0
        采样时间步长 dt（用于在求导时折算为真实物理单位）
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    s = pd.to_numeric(series, errors='coerce').copy()
    valid_mask = s.notna().to_numpy()
    n = len(s)
    
    if n < 4:
        return s
        
    # 1. 窗口长度奇数化与边界自适应修正
    w = int(window_length)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n if n % 2 != 0 else n - 1
    if w <= polyorder:
        polyorder = max(1, w - 1)
        
    # 2. 内部若有轻微缺失，先进行临时线性插值填平
    s_interpolated = s.interpolate(method='linear').ffill().bfill().to_numpy(dtype=float)
    
    # 3. 执行 S-G 滤波
    try:
        filtered = savgol_filter(
            s_interpolated, 
            window_length=w, 
            polyorder=polyorder, 
            deriv=deriv, 
            delta=delta, 
            mode='interp'
        )
    except Exception:
        filtered = s_interpolated
        
    res = pd.Series(filtered, index=series.index)
    col_name = str(series.name) if series.name else "signal"
    res.name = f"{col_name}_sg_d{deriv}"
    return res


def smooth_gaussian(
    series: pd.Series,
    sigma: float = 2.0
) -> pd.Series:
    """
    一维高斯核卷积平滑滤波。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors='coerce').interpolate().ffill().bfill().to_numpy(dtype=float)
    filtered = gaussian_filter1d(s, sigma=sigma, mode='nearest')
    return pd.Series(filtered, index=series.index, name=f"{series.name}_gauss")


def create_spatial_grid_2d(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    step: float
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    2D 空间搜寻/仿真区域栅格剖分器（生成离散网格与拓扑邻接字典）。
    
    参数：
    ----------
    x_bounds : (x_min, x_max)
        横向坐标范围（如经度或米）
    y_bounds : (y_min, y_max)
        纵向坐标范围（如纬度或米）
    step : float
        网格空间分辨率步长
        
    返回：
    ----------
    (pd.DataFrame, dict): 
        - DataFrame 包含每个网格编号、中心坐标与四至边界；
        - dict 映射每个网格周围四连通邻居的 ID 列表。
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    x_coords = np.arange(x_min, x_max, step)
    y_coords = np.arange(y_min, y_max, step)
    
    records = []
    grid_id = 0
    grid_map = {}
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            x_c = x + step / 2.0
            y_c = y + step / 2.0
            records.append({
                "grid_id": grid_id,
                "i_idx": i,
                "j_idx": j,
                "x_center": round(x_c, 4),
                "y_center": round(y_c, 4),
                "x_min": round(x, 4),
                "x_max": round(x + step, 4),
                "y_min": round(y, 4),
                "y_max": round(y + step, 4)
            })
            grid_map[(i, j)] = grid_id
            grid_id += 1
            
    grid_df = pd.DataFrame(records)
    
    # 构建 4 连通邻接关系
    adjacency = {}
    for (i, j), gid in grid_map.items():
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_coord = (i + di, j + dj)
            if neighbor_coord in grid_map:
                neighbors.append(grid_map[neighbor_coord])
        adjacency[gid] = neighbors
        
    return grid_df, adjacency


def create_spatial_grid_3d(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    z_bounds: Tuple[float, float],
    step: float
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    3D 水下/空域立体搜寻网格离散化。
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds
    
    x_coords = np.arange(x_min, x_max, step)
    y_coords = np.arange(y_min, y_max, step)
    z_coords = np.arange(z_min, z_max, step)
    
    records = []
    grid_id = 0
    grid_map = {}
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                records.append({
                    "grid_id": grid_id,
                    "x_center": round(x + step / 2.0, 4),
                    "y_center": round(y + step / 2.0, 4),
                    "z_center": round(z + step / 2.0, 4),
                })
                grid_map[(i, j, k)] = grid_id
                grid_id += 1
                
    grid_df = pd.DataFrame(records)
    
    # 6 连通邻居
    adjacency = {}
    for (i, j, k), gid in grid_map.items():
        neighbors = []
        for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nb = (i + di, j + dj, k + dk)
            if nb in grid_map:
                neighbors.append(grid_map[nb])
        adjacency[gid] = neighbors
        
    return grid_df, adjacency


def calc_distance_matrix(
    coords_a: Union[np.ndarray, pd.DataFrame],
    coords_b: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    metric: str = 'haversine'
) -> np.ndarray:
    """
    批量空间距离场矩阵计算器。
    
    参数：
    ----------
    coords_a : array-like, shape (N, 2)
        若 metric='haversine'，要求每行依次为 [纬度 Latitude, 经度 Longitude] (单位：度)；
        若 metric='euclidean'，为 [X, Y]
    coords_b : array-like, shape (M, 2), 可选
        若为 None，则计算 A 集合内部两两距离矩阵 (N, N)
    metric : str
        - 'haversine': 地球大圆球面真实距离（输出单位：公里 km）
        - 'euclidean': 平面欧氏距离
        - 'manhattan': 曼哈顿网格距离
    """
    a = np.asarray(coords_a, dtype=float)
    b = a if coords_b is None else np.asarray(coords_b, dtype=float)
    
    if metric == 'euclidean':
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))
        
    elif metric == 'manhattan':
        diff = np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :])
        return np.sum(diff, axis=-1)
        
    elif metric == 'haversine':
        # 经纬度转弧度
        lat1, lon1 = np.radians(a[:, 0]), np.radians(a[:, 1])
        lat2, lon2 = np.radians(b[:, 0]), np.radians(b[:, 1])
        
        dlat = lat2[np.newaxis, :] - lat1[:, np.newaxis]
        dlon = lon2[np.newaxis, :] - lon1[:, np.newaxis]
        
        hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1[:, np.newaxis]) * np.cos(lat2[np.newaxis, :]) * np.sin(dlon / 2.0) ** 2
        # 浮点截断防范
        hav = np.clip(hav, 0.0, 1.0)
        
        c = 2.0 * np.arcsin(np.sqrt(hav))
        r_earth = 6371.0 # 地球平均半径 (km)
        return r_earth * c
    else:
        raise ValueError(f"不支持的距离度量: {metric}，可选 'haversine', 'euclidean', 'manhattan'")


def plot_filter_comparison(
    raw_series: pd.Series,
    filtered_series: pd.Series,
    deriv_series: Optional[pd.Series] = None,
    title: str = "Savitzky-Golay Denoising & Numerical Differentiation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    绘制美赛出版级学术信号滤波前后对比图（含衍生的一阶导数/速度子图）。
    """
    n_plots = 2 if deriv_series is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, dpi=300 if save_path else 100)
    
    if n_plots == 1:
        axes = [axes]
        
    xs = np.arange(len(raw_series))
    
    # 1. 原始噪声信号 vs S-G 保形滤波信号
    axes[0].plot(xs, raw_series, color="#A6A6A6", alpha=0.7, linewidth=1.2, label="Raw Sensor Signal (Noisy)")
    axes[0].plot(xs, filtered_series, color="#1F4E79", linewidth=2.0, label="Savitzky-Golay Filtered")
    axes[0].set_ylabel("Amplitude / Position", fontsize=10)
    axes[0].set_title(title, fontsize=11, fontweight='bold')
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. 若提供导数信号，绘制速度图
    if deriv_series is not None:
        axes[1].plot(xs, deriv_series, color="#C00000", linewidth=1.8, label="Analytic 1st Derivative (Velocity)")
        axes[1].axhline(0, color="gray", linestyle=":", alpha=0.6)
        axes[1].set_ylabel("Velocity ($dx/dt$)", fontsize=10)
        axes[1].set_xlabel("Time Index / Sequence", fontsize=10)
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[0].set_xlabel("Time Index / Sequence", fontsize=10)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

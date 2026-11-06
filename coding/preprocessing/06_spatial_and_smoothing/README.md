# 06 - 空间几何网格化与信号滤波平滑 (Spatial & Smoothing)

本子模块提供用于空间连续区域网格化剖分与物理测量信号平滑的函数实现。涵盖 Savitzky-Golay 局部多项式保形滤波与解析数值求导、2D/3D 搜寻栅格生成及球面大圆距离（Haversine）矩阵计算。

> [!TIP]
> 局部多项式拟合原理、传感器差分噪声二次放大证明及网格搜索拓扑论述，请查阅：👉 [**原理、数值求导防散架与论文论证完全指南 (GUIDE.md)**](GUIDE.md)

---

## 快速调用示例

```python
import numpy as np
import pandas as pd
from spatial_smooth_toolkit import (
    smooth_savitzky_golay, 
    create_spatial_grid_2d, 
    calc_distance_matrix,
    plot_filter_comparison
)

# 1. 模拟含高频噪声的物理传感器轨迹
t = np.linspace(0, 10, 200)
noisy_signal = np.sin(t) + np.random.normal(0, 0.2, len(t))

# 2. Savitzky-Golay 保形平滑（窗口15，2阶多项式，保留局部极值点）
smooth_x = smooth_savitzky_golay(noisy_signal, window_length=15, polyorder=2, deriv=0)

# 3. 解析数值求导提取平滑速度曲线 (deriv=1，步长 dt=0.05)
velocity = smooth_savitzky_golay(noisy_signal, window_length=15, polyorder=2, deriv=1, delta=0.05)

# 4. 绘制滤波平滑前后对比图与速度导数曲线 (300 DPI)
plot_filter_comparison(noisy_signal, smooth_x, deriv_series=velocity, save_path="fig_sg_filter.png")

# 5. 2D 连续空间离散化为网格系统并构建四连通拓扑图
grid_df, adjacency_map = create_spatial_grid_2d(x_bounds=(0, 5), y_bounds=(0, 5), step=0.5)

# 6. 计算经纬度坐标集合两两之间的地球大圆真实距离矩阵 (km)
coords = np.array([
    [32.0, -117.0],
    [32.5, -117.5],
    [31.8, -118.0]
])
dist_matrix = calc_distance_matrix(coords, metric="haversine")
```

---

## 函数接口与参数说明

### 1. `smooth_savitzky_golay(series: pd.Series, window_length: int = 11, polyorder: int = 2, deriv: int = 0, delta: float = 1.0) -> pd.Series`
* **功能**：基于局部多项式最小二乘拟合的一维信号滤波器。
* **参数**：
  * `window_length`: 滑动窗口长度（必须为正奇数；若输入偶数，函数自动执行 $+1$ 修正；若超出序列长度则自动自适应截断）。
  * `polyorder`: 拟合多项式阶数，需满足 $\text{polyorder} < \text{window\_length}$。
  * `deriv`: 求导阶数。`0` 为平滑值；`1` 为平滑后的一阶导数（速度）；`2` 为平滑后的二阶导数（加速度）。直接解析求导避免了传统有限差分对高频噪声的二次放大效应（$\text{Var} \propto \Delta t^{-2}$）。
  * `delta`: 采样间隔时间步长 $\Delta t$。

### 2. `smooth_gaussian(series: pd.Series, sigma: float = 2.0) -> pd.Series`
* **功能**：一维高斯卷积平滑滤波，适合连续背景场或扩散过程。

### 3. `create_spatial_grid_2d(x_bounds: tuple, y_bounds: tuple, step: float) -> tuple[pd.DataFrame, dict]`
* **功能**：将指定的 2D 连续空间矩形区域划分为规则网格。
* **返回值**：
  * `grid_df`: 包含各网格编号、中心点坐标 $(x_c, y_c)$ 及上下界四至坐标的 DataFrame。
  * `adjacency_map`: 映射各网格 ID 与其四连通相邻网格 ID 列表的字典。

### 4. `create_spatial_grid_3d(x_bounds: tuple, y_bounds: tuple, z_bounds: tuple, step: float) -> tuple[pd.DataFrame, dict]`
* **功能**：3D 水下或空域立体网格剖分与六连通拓扑构建。

### 5. `calc_distance_matrix(coords_a: np.ndarray, coords_b: np.ndarray = None, metric: str = 'haversine') -> np.ndarray`
* **参数**：
  * `metric`:
    * `'haversine'`: 输入为 `[纬度, 经度]`（角度制），基于 Haversine 公式计算球面大圆真实距离，单位为公里（km）。内置浮点反余弦截断保护。
    * `'euclidean'`: 笛卡尔平面欧氏距离。
    * `'manhattan'`: 曼哈顿网格距离。

### 6. `plot_filter_comparison(raw_series: pd.Series, filtered_series: pd.Series, deriv_series: pd.Series = None, title: str = "...", save_path: str = None) -> plt.Figure`
* **功能**：绘制原始含噪信号与平滑后信号的时域轨迹对比图，可选绘制导数子图。

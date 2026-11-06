# 06 - 空间几何网格化与信号滤波平滑深度指南 (GUIDE)

> 本文档专为数学建模竞赛（MCM/ICM）队员编写，重点解答：**为什么带噪传感器数据直接数值差分是数学自杀？Savitzky-Golay 滤波器的多项式最小二乘原理是什么？**

---

## 目录
1. [数值微积分灾难：为什么传感器直接差分会导致噪声爆炸？](#一-数值微积分灾难为什么传感器直接差分会导致噪声爆炸)
2. [Savitzky-Golay 保形滤波与解析导数数学推导](#二-savitzky-golay-保形滤波与解析导数数学推导)
3. [空间离散网格化与拓扑搜寻建模](#三-空间离散网格化与拓扑搜寻建模)
4. [论文第 2/3 节：信号滤波与网格离散化英文论述模板](#四-论文第-23-节信号滤波与网格离散化英文论述模板)

---

## 一、 数值微积分灾难：为什么传感器直接差分会导致噪声爆炸？

在 B 题中，常见的情景是：题目给出某个运动目标（潜艇、无人机、生物个体）的离散位置坐标 $x_t$，要求建立动力学微分方程 $\frac{dx}{dt} = v_t$。

如果直接使用最简单的前向差分近似速度：
$$\hat{v}_t = \frac{x_{t+1} - x_t}{\Delta t}$$

假设真实位置受到高频随机噪声扰动 $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$：
$$\hat{x}_t = x_t + \varepsilon_t$$
则差分后的速度估计方差为：
$$\text{Var}(\hat{v}_t) = \frac{\text{Var}(\varepsilon_{t+1}) + \text{Var}(\varepsilon_t)}{\Delta t^2} = \frac{2\sigma^2}{\Delta t^2}$$

* **致命问题**：当采样间隔 $\Delta t$ 很小（例如高频采样 $\Delta t = 0.01\text{s}$）时，分母 $\Delta t^2 = 0.0001$，**噪声方差瞬间被放大了 20,000 倍**！
* 原始位置看起来只是一点点毛刺，差分算出的加速度会完全变成剧烈震荡的白噪声，后续微分方程求解直接发散爆炸。
* **破局之道**：**必须先通过 Savitzky-Golay 局部多项式拟合，再直接利用拟合多项式的解析导数计算速度与加速度！**

---

## 二、 Savitzky-Golay 保形滤波与解析导数数学推导

普通的移动平均（Moving Average）在滤波时把所有点等权平均，其代价是**把真实的极大值削低、把极小值填平（波峰畸变）**。

### S-G 局部多项式拟合原理：
对于以采样点 $i$ 为中心的窗口 $2m+1$ 个连续观测值，S-G 滤波器在窗口内拟合一个 $p$ 阶多项式（$p < 2m+1$）：
$$f(\tau) = a_0 + a_1 \tau + a_2 \tau^2 + \dots + a_p \tau^p, \quad \tau \in [-m, m]$$
使得窗口内均方误差最小化：
$$\min_{a_0, \dots, a_p} \sum_{\tau=-m}^m \left( f(\tau) - y_{i+\tau} \right)^2$$

* **直接获得平滑值**：取 $\tau = 0$ 处的多项式值，即 $f(0) = a_0$；
* **直接获得解析导数**：
  * 平滑速度：$v_i = \left. \frac{df}{d\tau} \right|_{\tau=0} \cdot \frac{1}{\Delta t} = \frac{a_1}{\Delta t}$；
  * 平滑加速度：$a_i = \left. \frac{d^2f}{d\tau^2} \right|_{\tau=0} \cdot \frac{1}{\Delta t^2} = \frac{2 a_2}{\Delta t^2}$。
* **数学性质**：系数可离线转化为卷积核卷积计算，速度极快；在有效压制随机白噪声的同时，**无偏保留了真实物理运动的曲率与加速度拐点**。

---

## 三、 空间离散网格化与拓扑搜寻建模

在搜索救援（如 2024 B 题潜艇）中，将连续的地理水域划分成为规整的 $2D / 3D$ 栅格节点是执行状态转移的基础：
1. **质心化表示**：每个网格以其中心坐标 $(x_c, y_c, z_c)$ 作为状态节点；
2. **拓扑连通图**：四连通（2D 平面）或六连通（3D 立体）构建图的边（Edge），赋予距离权重 $W_{ij} = \text{dist}(C_i, C_j)$；
3. **连通后续算法**：可直接输入到 Dijkstra 最短路径、马尔可夫决策过程（MDP）或蒙特卡洛搜寻模拟中。

---

## 四、 论文第 2/3 节：信号滤波与网格离散化英文论述模板

> **Section 2.5 Signal Denoising and Spatial Grid Discretization**  
> Raw kinematic trajectories recorded by underwater sensors inherently contain high-frequency instrumentation noise. Direct numerical finite differencing of raw coordinates would quadratically amplify measurement perturbations ($\text{Var}(\Delta x / \Delta t) \propto \Delta t^{-2}$), inducing divergent derivatives.  
> 
> To resolve this, a Savitzky-Golay local polynomial smoothing filter (window length $W = 15$, polynomial order $p = 2$) was implemented. By solving a localized least-squares polynomial approximation, the filter effectively suppresses high-frequency stochastic variance while rigorously preserving physical extrema and trajectory curvature. The smoothed velocity vector $\vec{v}_t$ was derived analytically from the first-order polynomial coefficient ($a_1 / \Delta t$), ensuring numerically stable inputs for the kinematic state equations.  
> 
> Furthermore, the continuous search basin was discretized into a 2D spatial grid network with spatial resolution $\Delta s = 0.5\text{ km}$, yielding $[N]$ discrete centroids and establishing four-connectivity adjacency topologies for subsequent Markovian search optimization.

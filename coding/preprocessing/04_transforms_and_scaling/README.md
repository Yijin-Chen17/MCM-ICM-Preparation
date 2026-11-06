# 04 - 分布矫正与指标无量纲化 (Transforms & Scaling)

本子模块提供用于分布形态正态化修正（Box-Cox 与 Yeo-Johnson 变换）与多属性综合评价（TOPSIS、熵权法）中的指标正向化与无量纲化映射算法。

> [!TIP]
> 各类正向化分段函数的数学推导与正态性变换的最优似然估计原理，请查阅：👉 [**原理、数学映射推导与论文公式块指南 (GUIDE.md)**](GUIDE.md)

---

## 快速调用示例

```python
import pandas as pd
from transform_toolkit import (
    transform_boxcox, 
    transform_yeojohnson, 
    scale_benefit, 
    scale_cost, 
    scale_interval, 
    normalize_topsis_vector,
    normalize_minmax_epsilon,
    plot_distribution_improvement
)

df = pd.read_csv("data.csv")

# 1. Box-Cox 变换（最大似然估计求解最优参数 lambda，自动处理非正数平移）
df["Income_Norm"], opt_lambda, shift_c = transform_boxcox(df["Income"])

# 2. 绘制核密度曲线与正态 Q-Q 图验证偏态改善效果
plot_distribution_improvement(df["Income"], df["Income_Norm"], save_path="fig_boxcox.png")

# 3. 效益型指标（极大型）归一化至 [0, 1]
df["Profit_Scaled"] = scale_benefit(df["Profit"])

# 4. 成本型指标（极小型）线性逆转至 [0, 1]
df["Emission_Reversed"] = scale_cost(df["Emission"], method="difference")

# 5. 区间最优型指标映射（处于最优区间内得分为 1，外部按线性惩罚递减）
df["Temp_Score"] = scale_interval(df["Temperature"], optimal_range=(20, 25), tolerance_range=(10, 35))

# 6. TOPSIS 专用向量模长规范化
df["Speed_Topsis"] = normalize_topsis_vector(df["Speed"])

# 7. 熵权法专用平移归一化（严格映射至 [eps, 1.0]，避免出现 ln(0) 数学未定义）
df["Index_Safe"] = normalize_minmax_epsilon(df["Index_A"], eps=1e-4)
```

---

## 函数接口与参数说明

### 1. `transform_boxcox(series: pd.Series, lmbda: float = None, shift_if_needed: bool = True) -> tuple[pd.Series, float, float]`
* **数学公式**：
  $$y^{(\lambda)} = \begin{cases} \frac{y^\lambda - 1}{\lambda}, & \lambda \ne 0 \\ \ln(y), & \lambda = 0 \end{cases}$$
* **机制**：若输入存在非正数且 `shift_if_needed=True`，自动执行平移 $y' = y + |y_{\min}| + 1.0$；通过 MLE 自动寻优最佳 $\hat{\lambda}$。

### 2. `transform_yeojohnson(series: pd.Series, lmbda: float = None) -> tuple[pd.Series, float]`
* **机制**：Yeo-Johnson 变换，支持零与负数输入，无需人工设定平移量。

### 3. `scale_benefit(series: pd.Series) -> pd.Series`
* **数学公式**：$x^* = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$。分母趋于 0 时返回全 0 向量。

### 4. `scale_cost(series: pd.Series, method: str = 'difference') -> pd.Series`
* **参数**：
  * `'difference'`: 线性极差逆转 $x^* = \frac{x_{\max} - x}{x_{\max} - x_{\min}}$（保持等距性）。
  * `'reciprocal'`: 倒数映射 $x^* = \frac{1/x - (1/x)_{\min}}{(1/x)_{\max} - (1/x)_{\min}}$。

### 5. `scale_interval(series: pd.Series, optimal_range: tuple, tolerance_range: tuple = None) -> pd.Series`
* **功能**：最佳区间型分段正向化。
* **参数**：
  * `optimal_range`: $(q_1, q_2)$ 为最适区间，内部得分为 1.0。
  * `tolerance_range`: $(a, b)$ 为容忍边界，区间外部按线性斜率单调惩罚至 0。

### 6. `normalize_topsis_vector(series: pd.Series) -> pd.Series`
* **数学公式**：$z_i = \frac{x_i}{\sqrt{\sum_{j=1}^n x_j^2}}$，满足向量模长归一化约束。

### 7. `normalize_minmax_epsilon(series: pd.Series, eps: float = 1e-4) -> pd.Series`
* **功能**：缩放映射至 $[\varepsilon, 1.0]$，用于熵权法中防止概率项 $p_i = 0$ 导致计算信息熵 $p_i \ln(p_i)$ 时溢出。

# 05 - 时序动态特征与动量建模 (Timeseries & Momentum)

本子模块提供针对离散事件序列与时间序列的特征工程算子，涵盖因果滑动窗口统计量提取、指数加权移动平均（EWMA）状态动量建模、连续状态运行长度（Streak）统计以及多阶滞后差分。

> [!TIP]
> 动量状态空间方程推导、记忆衰减半衰期设定依据及防止未来信息泄露准则，请查阅：👉 [**原理、势头数学建模与论文论述完全指南 (GUIDE.md)**](GUIDE.md)

---

## 快速调用示例

```python
import pandas as pd
from timeseries_toolkit import (
    calc_rolling_features, 
    calc_ewma_momentum, 
    calc_streak_length, 
    calc_lag_diff,
    plot_momentum_curve
)

# 模拟逐事件记录（例如点位胜负：1 为优势，-1 为劣势）
df = pd.DataFrame({
    "Point_Winner": [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1],
    "Is_Ace":       [0, 1, 0,  0, 0,  0,  0,  0, 1, 0, 1, 0]
})

# 1. 计算基于指数加权记忆衰减的动态动量状态指标 (alpha=0.3)
df["Momentum"] = calc_ewma_momentum(df["Point_Winner"], alpha=0.3)

# 2. 统计当前连续处于目标状态的持续长度 (Streak)
df["Win_Streak"] = calc_streak_length(df["Point_Winner"], target_value=1)

# 3. 多时间尺度滑动窗口统计特征衍生（均值、标准差、偏度）
df_rolling = calc_rolling_features(df["Point_Winner"], windows=[3, 5], metrics=["mean", "std", "skew"])

# 4. 计算滞后项 (Lag) 与一阶差分（变化速度）
df_lags = calc_lag_diff(df["Momentum"], lags=[1, 2], diffs=[1])

# 5. 绘制动量动态演化双轴曲线图（叠加标志性突发事件标记）
plot_momentum_curve(df.index, df["Momentum"], event_markers=df["Is_Ace"], save_path="fig_momentum.png")
```

---

## 函数接口与参数说明

### 1. `calc_rolling_features(series: pd.Series, windows: list = [3, 5, 10], metrics: list = ['mean', 'std', 'skew', 'max', 'min'], min_periods: int = 1) -> pd.DataFrame`
* **功能**：严格采用因果单向窗口（仅基于历史与当前观测计算，无前向信息引入），生成多时间尺度的移动矩与极值指标。
* **参数**：
  * `windows`: 整数列表，定义不同滑动窗口长度。
  * `min_periods`: 窗口内最小观测数。默认设为 1，防止序列起始阶段产生整段 NaN。

### 2. `calc_ewma_momentum(series: pd.Series, alpha: float = 0.3, span: float = None, halflife: float = None) -> pd.Series`
* **数学公式**：
  $$M_t = \alpha Y_t + (1 - \alpha) M_{t-1}$$
* **机制**：通过指数加权递推量化“即时事件冲击”与“历史状态衰减”的合成动态，参数 `alpha` 决定状态更新速率；亦支持通过 `halflife`（半衰期）指定记忆衰退周期。

### 3. `calc_streak_length(series: pd.Series, target_value: int | float | str = 1) -> pd.Series`
* **功能**：计算连续处于 `target_value` 状态的累积长度。遇到非目标值即刻归零。纯向量化遍历，无额外性能开销。

### 4. `calc_lag_diff(series: pd.Series, lags: list = [1, 2], diffs: list = [1, 2]) -> pd.DataFrame`
* **功能**：生成指定阶数的自回归滞后项 $X_{t-k}$ 与差分项 $\Delta^d X_t = X_t - X_{t-d}$，捕捉变化趋势与加速度。

### 5. `resample_regular_series(df: pd.DataFrame, time_col: str, freq: str = '1D', agg_dict: dict = None, fill_method: str = 'ffill') -> pd.DataFrame`
* **功能**：将不规则离散时间戳规整为等间隔采样序列，支持指定聚合字典与补空机制。

### 6. `plot_momentum_curve(time_series: pd.Series, momentum_series: pd.Series, event_markers: pd.Series = None, save_path: str = None) -> plt.Figure`
* **功能**：生成双轴学术图表，以上下色块分区展示优势期与劣势期，并对齐主轴事件标记。

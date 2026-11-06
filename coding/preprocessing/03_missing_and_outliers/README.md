# 03 - 缺失与异常精细治理 (Missing & Outliers)

本子模块提供用于离群值识别、缩尾治理（Winsorization）与缺失值插补的算法实现。强调在不牺牲样本容量的前提下控制极端值方差，并针对连续物理量提供保形插值方法。

> [!TIP]
> 缩尾处理的统计学论据、PCHIP 与三次样条插值的数学性质对比，请查阅：👉 [**原理、插值选型与论文辩护完全指南 (GUIDE.md)**](GUIDE.md)

---

## 快速调用示例

```python
import pandas as pd
from cleaning_toolkit import (
    detect_outliers_iqr, 
    winsorize_iqr, 
    winsorize_quantiles, 
    interpolate_series, 
    impute_knn,
    plot_outlier_inspection
)

df = pd.read_csv("data.csv")

# 1. Tukey IQR 准则离群点检测（返回布尔掩码）
mask_outlier = detect_outliers_iqr(df["Speed"], k=1.5)
print(f"检出异常值数量: {mask_outlier.sum()}")

# 2. 缩尾处理（将离群极值投影回四分位截断边界，保持样本总数不变）
df["Speed_Winsorized"] = winsorize_iqr(df["Speed"], k=1.5)

# 3. 绘制治理前后箱线图与核密度分布对比图 (300 DPI)
plot_outlier_inspection(df["Speed"], df["Speed_Winsorized"], save_path="fig_winsorize.png")

# 4. 分段三次埃尔米特保形插值（保持局部单调性，防止样条插值出现虚假振荡与负值）
df["Temperature_Clean"] = interpolate_series(df["Temperature"], method="pchip")

# 5. 基于辅助特征欧氏空间距离的 KNN 缺失插补
df["Score_Imputed"] = impute_knn(df, target_col="Score", feature_cols=["Age", "Salary", "Experience"], n_neighbors=5)
```

---

## 函数接口与参数说明

### 1. `detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series`
* **功能**：依据 Tukey 准则判定离群值。
* **判定阈值**：超出 $[Q_1 - k \cdot \text{IQR}, Q_3 + k \cdot \text{IQR}]$ 区间的样本标记为 `True`。参数 $k=1.5$ 对应中度离群，$k=3.0$ 对应极端离群。

### 2. `detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series`
* **功能**：依据样本标准化绝对值度量偏离度，适合近似正态分布的数据。

### 3. `winsorize_iqr(series: pd.Series, k: float = 1.5) -> pd.Series`
* **功能**：双端 Tukey 缩尾截断。低于下限投影至下边界，高于上限投影至上边界。保留原数据所有行，不降低样本容量与统计检验自由度。

### 4. `winsorize_quantiles(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series`
* **功能**：基于经验分位数的双端截断，适合极度厚尾的金融与网络流量指标。

### 5. `interpolate_series(series: pd.Series, method: str = 'pchip') -> pd.Series`
* **功能**：一维连续数据插补。
* **参数**：
  * `'pchip'`: 分段三次埃尔米特保形插值（Piecewise Cubic Hermite Interpolating Polynomial）。严格保持插值区间的局部单调性，不引入虚假极值，为物理量与浓度数据首选。
  * `'linear'`: 线性插值。
  * `'spline'`: 三次样条插值。

### 6. `impute_knn(df: pd.DataFrame, target_col: str, feature_cols: list, n_neighbors: int = 5) -> pd.Series`
* **功能**：基于标准化特征空间中的欧几里得距离，提取距离最近的 $K$ 个完整样本，按距离倒数加权平均填补目标列缺失值。

### 7. `plot_outlier_inspection(series_raw: pd.Series, series_clean: pd.Series = None, title: str = "...", save_path: str = None) -> plt.Figure`
* **功能**：生成两子图并排的箱线图（Boxplot）与核密度估计曲线（KDE），展示截断治理前后的离散度变化。

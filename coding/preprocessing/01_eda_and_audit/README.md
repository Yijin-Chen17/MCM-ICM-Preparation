# 01 - 数据体检与描述性统计 (EDA & Audit)

本子模块提供用于探索性数据分析（EDA）与数据质量审计的函数工具，支持数据基本分布参数提取、高阶统计矩（偏度、峰度）计算、常数列识别以及 LaTeX 格式 Table 1 描述性统计三线表导出。

> [!TIP]
> 理论推导、指标判定阈值及论文实证英文模板，请查阅：👉 [**原理、指标判定与论文描述完全指南 (GUIDE.md)**](GUIDE.md)

---

## 快速调用示例

```python
import pandas as pd
from eda_toolkit import audit_dataset, export_latex_table1, plot_correlation_matrix, plot_missing_matrix

df = pd.read_csv("data.csv")

# 1. 执行全字段数据审计（返回包含类型、缺失率、分位数、偏度、峰度的 DataFrame）
audit_report = audit_dataset(df)
print(audit_report[["Column", "Missing_Rate(%)", "Mean", "Std", "Skewness", "Audit_Note"]])

# 2. 导出 LaTeX Table 1 描述性统计三线表代码
latex_code = export_latex_table1(df, caption="Descriptive Statistics of Experimental Data", precision=2)
print(latex_code)

# 3. 绘制相关性热力图（可设定阈值过滤弱相关项）
plot_correlation_matrix(df, method='pearson', threshold=0.2, save_path="fig_corr.png")

# 4. 绘制缺失值空间分布矩阵
plot_missing_matrix(df, save_path="fig_missing.png")
```

---

## 函数接口与参数说明

### 1. `audit_dataset(df: pd.DataFrame, skew_threshold: float = 1.0, kurt_threshold: float = 3.0) -> pd.DataFrame`
* **功能**：遍历各列，计算缺失率、唯一值基数、均值、标准差、中位数、四分位距（IQR）、样本偏度（Fisher-Pearson 偏度）与超额峰度（Fisher 峰度）。
* **容错机制**：滤除 `inf` / `-inf` 异常浮点值；对全空列与零方差列自动打标 `Zero Variance` 或 `High Missing`。

### 2. `export_latex_table1(df: pd.DataFrame, columns: list = None, caption: str = "...", label: str = "tab:data_summary", precision: int = 3) -> str`
* **功能**：生成符合 Booktabs 规范的 LaTeX 三线表源码。
* **参数**：
  * `columns`: 指定输出的数值列名列表；默认 `None` 时自动选择全部数值列。
  * `precision`: 浮点数输出格式化精度位数。
* **容错机制**：自动转义字段名中包含的特殊字符（如 `_`, `%`, `$`, `&`），确保 LaTeX 编译器兼容性。

### 3. `plot_correlation_matrix(df: pd.DataFrame, columns: list = None, method: str = 'pearson', threshold: float = None, save_path: str = None) -> plt.Figure`
* **功能**：计算并可视化自变量间相关系数矩阵。
* **参数**：
  * `method`: 支持 `'pearson'`（线性相关）与 `'spearman'`（秩相关）。
  * `threshold`: 过滤绝对值小于该阈值的非对角线系数值，突出强相关特征。

### 4. `detect_constant_columns(df: pd.DataFrame, drop_na: bool = True) -> list[str]`
* **功能**：检测数据集中方差为 0 的常数列或全缺失列，返回列名列表供前置特征筛选。

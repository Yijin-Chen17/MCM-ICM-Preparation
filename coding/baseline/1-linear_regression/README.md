# 01 - 线性回归基准模型脚手架 (Linear Regression Baseline)

本模版专为美国大学生数学建模竞赛（MCM/ICM）设计，用于在数据清洗完成后快速构建**可解释性强、数理统计检验完备的线性回归基准模型（Baseline）**。

> [!TIP]
> **关于参数选型依据、统计检验推导与论文实证分析模板：**  
> 请查阅同目录下的指南文档：👉 [**原理、选型与结果解读指南 (GUIDE.md)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/baseline/1-linear_regression/GUIDE.md)

---

## 一、 快速调用示例

```python
import pandas as pd
from baseline.linear_regression import MCMRegressionBaseline

df = pd.read_csv("cleaned_data.csv")
baseline = MCMRegressionBaseline(df, target_col="Target_Y").fit()

baseline.summary()
baseline.plot_diagnostics(save_dir="figures/", combine_4in1=True)
print(baseline.to_latex_table())
```

---

## 二、 外部可调用函数完整签名与参数清单

### 1. 初始化类：`MCMRegressionBaseline(...)`

```python
MCMRegressionBaseline(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list = None,
    split_data: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    test_df: pd.DataFrame = None,
    feature_scaling: str = 'zscore',
    target_transform: str = None,
    log_features: list = None,
    add_polynomial: bool = False,
    add_interactions: bool = False
)
```

#### 重要参数与可选值清单：

* **`feature_scaling`** 可选值：
  * `'zscore'` (默认)
  * `'minmax'`
  * `'robust'`
  * `'centering'`
  * `None`

* **`target_transform`** 可选值：
  * `None` (默认)
  * `'log'`
  * `'boxcox'`
  * `'sqrt'`

* **其他参数格式说明**：
  * `df`: 必须填，`pd.DataFrame` 格式
  * `target_col`: 必须填，目标列字符串，如 `"Score"`
  * `feature_cols`: 默认 `None` 自动提取全部数值列；手动指定如 `['GDP', 'Pop']`
  * `split_data`: `False` (默认全量拟合) / `True` (内部切分)
  * `test_size`: 切分比例，如 `0.2`
  * `random_state`: 随机种子，如 `42`
  * `test_df`: 外部测试集，默认 `None`；如 `test_data`
  * `log_features`: 指定特征取对数，默认 `None`；如 `['GDP', 'Pop']`
  * `add_polynomial`: `False` (默认) / `True` (增加 $X^2$)
  * `add_interactions`: `False` (默认) / `True` (增加 $X_i \cdot X_j$)

---

### 2. 模型训练：`.fit()`

```python
baseline.fit()
```
* **参数**：无。
* **输出**：返回拟合后的自身实例对象（`self`）。

---

### 3. 模型诊断报告：`.summary()`

```python
baseline.summary()
```
* **参数**：无。
* **输出**：在终端打印三部分内容：
  1. 模型指标（$R^2$、Adj $R^2$、RMSE、MAE、$F$ 统计量与 $p$ 值）
  2. 统计学假设检验（Durbin-Watson、Breusch-Pagan $p$ 值、Jarque-Bera $p$ 值、最大 VIF）
  3. 各特征 VIF 降序排列完整表格

---

### 4. 统计检验字典提取：`.diagnose()`

```python
results = baseline.diagnose()
```
* **参数**：无。
* **输出**：返回一个 `dict` 字典，包含 DW 统计量、BP 检验 $p$ 值、JB 检验 $p$ 值、最大 VIF 值。

---

### 5. 单独出图函数族

#### ① 真实值 vs 预测值散点图
```python
baseline.plot_actual_vs_predicted(save_path: str = None)
```
* **参数**：`save_path="pred.png"`（不填则仅预览不保存）。
* **输出**：散点图（含 $y=x$ 参考线，已还原真实物理量纲）。

#### ② 残差 vs 拟合值散点图
```python
baseline.plot_residuals_vs_fitted(save_path: str = None)
```
* **参数**：`save_path="resid.png"`（可选）。
* **输出**：散点图（含 $y=0$ 基准线，用于检验同方差性）。

#### ③ 正态 Q-Q 图
```python
baseline.plot_qq(save_path: str = None)
```
* **参数**：`save_path="qq.png"`（可选）。
* **输出**：分位数图（用于检验残差正态性）。

#### ④ 特征重要性条形图
```python
baseline.plot_feature_importance(save_path: str = None)
```
* **参数**：`save_path="importance.png"`（可选）。
* **输出**：各特征标准化回归系数水平条形图（红绿区分正负影响，降序排列）。

---

### 6. 统一批量出图控制：`.plot_diagnostics(...)`

```python
baseline.plot_diagnostics(
    save_dir: str = None, 
    combine_4in1: bool = False, 
    show: bool = True
)
```
* **参数选项**：
  * `save_dir`: 保存路径，如 `"figures/"`。默认 `None` 时仅在窗口显示。若传入路径，自动分别保存 4 张单图。
  * `combine_4in1`: `False` (默认，仅保存单图) / `True` (额外拼合输出一张 2x2 学术大图 `diagnostic_4in1.png`)。
  * `show`: `True` (默认弹窗预览) / `False` (静默运行仅保存)。

---

### 7. 导出 LaTeX 学术三线表：`.to_latex_table()`

```python
latex_code = baseline.to_latex_table()
```
* **参数**：无。
* **输出**：返回一段标准 Booktabs 规范的 LaTeX 三线表源码字符串（含 `*`, `**`, `***` 显著性标注）。

---

### 8. 常用数据属性直接读取

* `baseline.metrics`：评估指标字典（`dict`）
* `baseline.vif_df`：VIF 特征共线性表格（`pd.DataFrame`）
* `baseline.model`：底层 `statsmodels` 拟合结果模型对象

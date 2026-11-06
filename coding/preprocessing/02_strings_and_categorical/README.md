# 02 - 字符串清洗与类别型特征编码 (Strings & Categorical)

本子模块提供用于非数值字段清洗与类别型特征数值化编码的函数工具，涵盖文本型数值解析、二元变量映射、保序等级编码、带低频归并的独热编码（One-Hot）以及无信息泄露的目标编码（Target Encoding）。

> [!TIP]
> 虚拟变量陷阱（Dummy Variable Trap）的数学防范与平滑目标编码推导，请查阅：👉 [**原理、编码防灾与论文对照表规范 (GUIDE.md)**](GUIDE.md)

---

## 快速调用示例

```python
import pandas as pd
from category_toolkit import (
    clean_numeric_string, 
    encode_binary, 
    encode_ordinal, 
    encode_onehot, 
    encode_target_cv, 
    export_codebook_latex
)

df = pd.DataFrame({
    "Price": ["$1,200.50", "$3,500.00", "(150.00)", "N/A", "85.2%"],
    "Result": ["Win", "Loss", "Win", "Win", "Loss"],
    "Tier": ["Low", "High", "Medium", "Low", "High"],
    "Surface": ["Hard", "Grass", "Clay", "Carpet", "Hard"],
    "Target_Y": [1.2, 3.4, 0.8, 1.5, 2.9]
})

# 1. 移除非数字符号并解析为标准浮点型
df["Price_Clean"] = clean_numeric_string(df["Price"])

# 2. 二元类别映射为 0/1 数值
df["Result_Code"] = encode_binary(df["Result"])

# 3. 有序等级分类映射为离散整数
df["Tier_Code"] = encode_ordinal(df["Tier"], order_list=["Low", "Medium", "High"])

# 4. 独热编码（频次低于阈值的长尾类别自动并入 'Other'，支持 drop_first 防共线性）
df_encoded = encode_onehot(df, column="Surface", min_freq=0.25, drop_first=True)

# 5. 基于 K 折交叉验证的目标均值平滑编码（防止数据泄露）
df["Surface_Target"] = encode_target_cv(df, cat_col="Surface", target_col="Target_Y", n_splits=3)

# 6. 导出变量编码对照 LaTeX 三线表
codebook = {
    "Result": {"Loss": 0, "Win": 1},
    "Tier": {"Low": 0, "Medium": 1, "High": 2}
}
print(export_codebook_latex(codebook))
```

---

## 函数接口与参数说明

### 1. `clean_numeric_string(series: pd.Series, strip_chars: list = None, default_val: float = np.nan) -> pd.Series`
* **功能**：通过正则表达式滤除货币符号（如 `$`, `¥`, `€`）、千分位逗号、百分号及物理单位，支持将括号表示的会计负数（如 `(150.0)`）转换为标准负数 `-150.0`。
* **参数**：
  * `strip_chars`: 额外需要精确移除的特殊文本单元列表。
  * `default_val`: 无法解析字段的替代值。

### 2. `encode_binary(series: pd.Series, mapping: dict = None, default_val: float = np.nan) -> pd.Series`
* **功能**：二元分类变量映射为数值型。
* **参数**：
  * `mapping`: 显式映射字典（如 `{'Win': 1, 'Loss': 0}`）。若设为 `None`，则自动检测序列中的唯二非空项并按频次排序映射为 1 与 0。

### 3. `encode_ordinal(series: pd.Series, order_list: list, default_val: float = np.nan) -> pd.Series`
* **功能**：将具备明确等级顺序的分类标签映射为单调递增的非负整数。
* **参数**：
  * `order_list`: 严格从低到高排列的类别名称序列。

### 4. `encode_onehot(df: pd.DataFrame, column: str, min_freq: float = 0.03, other_label: str = 'Other', drop_first: bool = False, prefix: str = None) -> pd.DataFrame`
* **功能**：对指定分类列执行哑变量生成。
* **参数**：
  * `min_freq`: 类别样本频率截断阈值。频率占比低于此阈值的离散项将被统一替换为 `other_label`，防止生成过度稀疏的高维矩阵。
  * `drop_first`: 是否剔除第一个基准水平，防止线性模型出现多重共线性（Dummy Variable Trap）。

### 5. `encode_target_cv(df: pd.DataFrame, cat_col: str, target_col: str, n_splits: int = 5, smoothing: float = 10.0, random_state: int = 42) -> pd.Series`
* **功能**：高基数分类变量的目标平滑编码。
* **数学原理**：利用 Out-of-fold（K 折交叉）机制在子样本内计算贝叶斯经验平滑均值，严格保证验证集样本未被当前折计算包含，杜绝特征穿越（Data Leakage）。

### 6. `export_codebook_latex(mapping_dict: dict, caption: str = "...", label: str = "tab:codebook") -> str`
* **功能**：自动生成变量编码定义的 LaTeX Table 三线表代码。

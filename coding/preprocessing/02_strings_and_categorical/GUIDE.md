# 02 - 字符串清洗与类别型转数值完全指南 (GUIDE)

> 本文档专为数学建模竞赛（MCM/ICM）队员编写，重点解答：**独热编码的三大死穴、目标编码防止数据泄露的数学原理，以及如何在论文中规范展示变量对照表**。

---

## 目录
1. [独热编码（One-Hot）的三大陷阱与防范](#一-独热编码one-hot的三大陷阱与防范)
   - [1.1 虚拟变量陷阱（Dummy Variable Trap）与共线性](#11-虚拟变量陷阱dummy-variable-trap与共线性)
   - [1.2 维度爆炸与稀疏性灾难（长尾截断机制）](#12-维度爆炸与稀疏性灾难长尾截断机制)
2. [高基数特征的目标平滑编码（Target Encoding）数学推导](#二-高基数特征的目标平滑编码target-encoding数学推导)
3. [论文第 2 节：变量编码说明写作规范与 LaTeX Codebook](#三-论文第-2-节变量编码说明写作规范与-latex-codebook)

---

## 一、 独热编码（One-Hot）的三大陷阱与防范

### 1.1 虚拟变量陷阱（Dummy Variable Trap）与共线性
在线性回归、Logistic 回归及任何基于矩阵求逆（$(X^T X)^{-1}$）的模型中，如果一个包含 $K$ 个水平的分类变量被编码为 $K$ 个独热虚拟变量，且模型包含截距项 $\beta_0$，则由于：
$$\sum_{j=1}^K D_j = 1$$
虚拟变量之和恒等于截距常数向量，导致设计矩阵 $X$ **严格多重共线性（Rank Deficient）**。
* **后果**：回归系数无法唯一定解，VIF 趋于无穷大，显著性 $p$ 值完全失效。
* **规避准则**：在配合线性基准模型（如 `baseline/1-linear_regression`）时，**必须开启 `drop_first=True`**，仅保留 $K-1$ 个虚拟列，以剔除的那个类别作为基准参考组（Reference Level）。

---

### 1.2 维度爆炸与稀疏性灾难（长尾截断机制）
在很多实际赛题中，某一列类别（如“选手出生城市”、“商品细分类别”）可能包含几十甚至上百个唯一值。如果机械地使用全部独热，列数瞬间暴增几十倍，导致：
1. 样本被过度稀疏化（大部分行全是 0）；
2. 树模型划分准则严重偏向于高基数类别，引发灾难性过拟合。

**解决策略（本工具箱的核心实现）**：
设置出现频率阈值（例如 `min_freq=0.03`）。仅对样本占比 $\ge 3\%$ 的主要类别独立独热；所有出现极少的长尾小类别统一归纳为 `"Other"`，仅生成一列 `prefix_Other`。

---

## 二、 高基数特征的目标平滑编码（Target Encoding）数学推导

当类别数量多达上百个且不能直接独热时（如 30 个参赛队伍），**目标编码**是最佳手段。但若直接用类别内目标变量的简单均值替换，会引发严重的**训练集数据泄露（Overfitting Leakage）**。

本工具箱采用**交叉验证经验贝叶斯平滑（Empirical Bayes Smoothing with Out-of-Fold CV）**：
对于类别 $c$ 中的样本，其编码值 $S_c$ 计算为：
$$S_c = \frac{n_c \cdot \bar{y}_c + m \cdot \mu_{\text{global}}}{n_c + m}$$

其中：
* $n_c$：当前折训练集内属于类别 $c$ 的样本数量；
* $\bar{y}_c$：当前折训练集内类别 $c$ 对应目标变量的条件均值；
* $\mu_{\text{global}}$：整体训练集的全局目标均值；
* $m$：平滑权重（默认 $m=10$）。

**数学直觉**：
* 当该类别样本量极少时（$n_c \to 0$），估算均值不可靠，平滑值自动收缩至全局均值 $\mu_{\text{global}}$；
* 当该类别样本量充足时（$n_c$ 极大），平滑值逼近其真实的组内均值 $\bar{y}_c$；
* 配合 5 折交叉验证（Out-of-Fold），当前样本的编码值**绝不包含其自身的目标标签**，数学上杜绝特征穿越。

---

## 三、 论文第 2 节：变量编码说明写作规范与 LaTeX Codebook

在论文中，不能让评委去猜 `0` 代表什么、`1` 代表什么，必须附上规范的变量定义表：

> **Section 2.2 Categorical Feature Harmonization and Encoding**  
> Qualitative variables were mathematically mapped into numerical representations to facilitate subsequent modeling. Binary indicators (e.g., match outcomes) were mapped to $\{0, 1\}$. For ordinal variables, monotonicity was preserved via ranked ordinal mapping. To prevent multicollinearity (the dummy variable trap) in baseline regression formulations, one-hot encoding was implemented with reference category dropping. Furthermore, sparse categorical levels with empirical occurrences below $3\%$ were aggregated into an `"Other"` category to circumvent excessive dimensionality. The complete numerical mapping codebook is presented in Table 2.

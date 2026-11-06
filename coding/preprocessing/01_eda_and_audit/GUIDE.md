# 01 - 数据体检与描述性统计深度指南 (GUIDE)

> 本文档专为数学建模竞赛（MCM/ICM）队员编写，重点解答：**为什么论文必须做数据体检、偏度峰度怎么判定、相关性矩阵怎么写分析，以及论文第 2 节的完整英文论述模板**。

---

## 目录
1. [为什么美赛评委看重数据概况与体检报告？](#一-为什么美赛评委看重数据概况与体检报告)
2. [核心统计指标判定法则（怎么看诊断结果？）](#二-核心统计指标判定法则怎么看诊断结果)
   - [2.1 偏度 (Skewness) 判定与处理建议](#21-偏度-skewness-判定与处理建议)
   - [2.2 峰度 (Kurtosis) 判定与厚尾风险](#22-峰度-kurtosis-判定与厚尾风险)
   - [2.3 缺失率与常数列过滤准则](#23-缺失率与常数列过滤准则)
3. [论文第 2 节：标准英文论述与 Table 1 排版模板](#三-论文第-2-节标准英文论述与-table-1-排版模板)

---

## 一、 为什么美赛评委看重数据概况与体检报告？

在 MCM/ICM 评审中，评委首先会审阅摘要，接着快速扫视 **Section 2: Data Description & Preprocessing**。
- **坏论文做法**：没有任何数据统计总览，开篇直接放复杂的算法公式，或者草草写一句 "We cleaned the data and removed missing values"。评委无法评估数据真实质量，会质疑后续模型的可靠性。
- **O 奖/高分论文做法**：开篇给出规范的 **Table 1: Descriptive Statistics and Data Health Summary**，清晰列出样本量、均值、分位数、缺失率与偏度，并向评委证明：“我们已充分了解数据的分布形态与异质性，后续的所有变换均有严密的统计学依据”。

---

## 二、 核心统计指标判定法则（怎么看诊断结果？）

### 2.1 偏度 (Skewness) 判定与处理建议
偏度度量了数据分布的不对称程度（基于三阶中心矩）：
$$S = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^3}{\left(\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right)^{3/2}}$$

| 偏度取值范围 | 分布形态 | 现实含义 | 推荐应对措施 |
| :--- | :--- | :--- | :--- |
| **$[-0.5, 0.5]$** | 近似对称 (Symmetric) | 数据接近正态，可以直接用于回归、PCA 等算法 | 无需特殊处理，直接使用 |
| **$[0.5, 1.0]$ 或 $[-1.0, -0.5]$** | 中度偏态 (Moderately Skewed) | 存在一定长尾或拖尾 | 考虑 Square Root ($\sqrt{X}$) 或温和缩放 |
| **$> 1.0$ (正偏/右偏)** | **严重右偏 (Highly Right-Skewed)** | 极少数极大值拉长了均值（如财富、感染人数、碳排放量、击球时长） | **强烈建议采用对数变换 $\ln(X+1)$ 或 Box-Cox 变换** |
| **$< -1.0$ (负偏/左偏)** | **严重左偏 (Highly Left-Skewed)** | 极小值拖尾（如成熟产品的合格率、高难度考试分数分布） | 采用反射对数或 Yeo-Johnson 变换 |

---

### 2.2 峰度 (Kurtosis) 判定与厚尾风险
本工具计算的是 **Fisher 峰度**（正态分布基准为 0）：
$$\text{Kurt} = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^4}{\left(\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right)^2} - 3$$

* **$\text{Kurt} \approx 0$ (Mesokurtic)**：分布陡峭程度与标准正态一致。
* **$\text{Kurt} > 3.0$ (Leptokurtic / Heavy-tailed)**：**尖峰厚尾**。说明数据中心极其尖锐，但两端存在大量超出正常正态波动的极端事件（极端天气、金融崩盘、比赛极端逆转）。
  * **建议**：在此类特征上**绝对不要直接使用普通均值或欧氏距离**，应配合鲁棒缩放器（RobustScaler）或缩尾处理（Winsorization）。

---

### 2.3 缺失率与常数列过滤准则
* **缺失率 $> 50\%$**：若该特征不是赛题核心研究变量，通常应考虑剔除该字段，或仅生成二元标记列（Missing Indicator: 0=未缺失, 1=缺失）。
* **唯一值数 $\le 1$（Zero Variance）**：信息量为 0，输入任何模型只会增加自由度负担甚至导致矩阵奇异，必须无条件删除。

---

## 三、 论文第 2 节：标准英文论述与 Table 1 排版模板

直接复制以下英文模板至美赛论文，稍作变量替换即可：

> **Section 2.1 Data Overview and Statistical Exploration**  
> Prior to model formulation, a comprehensive exploratory data analysis (EDA) was performed across all $N = [Insert Count]$ samples to examine distributional properties and potential data anomalies. Table 1 summarizes the primary statistical descriptors, including central tendencies, dispersion metrics, missingness ratios, and higher-order moments.  
> 
> As revealed in Table 1, the variable *[Insert Variable 1]* exhibits pronounced positive skewness ($S = [Value] > 1.0$) and leptokurtic behavior ($\text{Kurt} = [Value]$), indicating severe right-tail concentration. Conversely, the feature *[Insert Variable 2]* conforms well to an approximately symmetric distribution ($|S| < 0.5$). For variables demonstrating substantial skewness and heavy tails, specific normalization transformations will be applied in Section 2.2 to prevent algorithmic instability.

# 04 - 分布正态化与评价无量纲化深度指南 (GUIDE)

> 本文档专为数学建模竞赛（MCM/ICM）队员编写，重点解答：**为什么综合评价不能直接求和？四大指标正向化映射的数学证明，以及 Box-Cox 变换的论文公式书写规范**。

---

## 目录
1. [综合评价指标方向性（正向化）数学推导](#一-综合评价指标方向性正向化数学推导)
   - [1.1 极小型指标（成本型）映射公式](#11-极小型指标成本型映射公式)
   - [1.2 区间型指标（居中最佳型）分段惩罚映射](#12-区间型指标居中最佳型分段惩罚映射)
2. [Box-Cox 与 Yeo-Johnson 幂变换理论](#二-box-cox-与-yeo-johnson-幂变换理论)
3. [论文第 2/3 节：正向化与无量纲化英文排版模板](#三-论文第-23-节正向化与无量纲化英文排版模板)

---

## 一、 综合评价指标方向性（正向化）数学推导

在多属性决策模型（TOPSIS、熵权法、VIKOR、灰度关联度等）中，不同维度的指标往往存在**极性相悖（Directional Conflict）**与**量纲不一**的问题。必须经过严格的正向化与无量纲化处理。

### 1.1 极小型指标（成本型）映射公式
对于成本型指标 $X$（数值越小方案越优，如失误率、污染物排放量），推荐采用**差值极大极小映射**：
$$x_i^* = \frac{\max(X) - x_i}{\max(X) - \min(X)}$$
* **数学性质**：严格将原始最小值映射为满分 1.0，最大值映射为零分 0.0，且完美保持原始数值之间的等距线性关系。

---

### 1.2 区间型指标（居中最佳型）分段惩罚映射
某些指标并非单调越大约好，而是落在特定区间 $[q_1, q_2]$ 时最理想（如人体舒适温度 $20 \sim 25^\circ\text{C}$，水体溶氧量等）。给定容忍界限 $[a, b]$（其中 $a < q_1 < q_2 < b$），分段正向化函数构造为：
$$x_i^* = \begin{cases}
1 - \frac{q_1 - x_i}{q_1 - a}, & a \le x_i < q_1 \\
1, & q_1 \le x_i \le q_2 \\
1 - \frac{x_i - q_2}{b - q_2}, & q_2 < x_i \le b \\
0, & x_i < a \text{ or } x_i > b
\end{cases}$$

---

## 二、 Box-Cox 与 Yeo-Johnson 幂变换理论

### Box-Cox 变换公式
对于原始严格为正的特征 $Y > 0$（若有零或负值，先执行正向平移 $Y' = Y + c$）：
$$Y^{(\lambda)} = \begin{cases}
\frac{Y^\lambda - 1}{\lambda}, & \text{if } \lambda \ne 0 \\
\ln(Y), & \text{if } \lambda = 0
\end{cases}$$
算法通过**最大似然估计（MLE）**在实数轴上自动搜寻让变换后残差对数似然函数最大的参数 $\hat{\lambda}$。

---

## 三、 论文第 2/3 节：正向化与无量纲化英文排版模板

> **Section 2.4 Indicator Directional Alignment and Normalization**  
> To resolve dimensionality discrepancies and directional incongruities among heterogeneous evaluation criteria, systematic forward transformations were conducted prior to comprehensive scoring:  
> 1. **Benefit-type attributes** were normalized via standard linear Min-Max scaling:  
>    $$x_{ij}^* = \frac{x_{ij} - \min_i(x_{ij})}{\max_i(x_{ij}) - \min_i(x_{ij})}$$  
> 2. **Cost-type attributes** (e.g., energy consumption) were reversed such that lower empirical values correspond to higher relative scores:  
>    $$x_{ij}^* = \frac{\max_i(x_{ij}) - x_{ij}}{\max_i(x_{ij}) - \min_i(x_{ij})}$$  
> 3. **Interval-optimal features** (e.g., ideal operating temperature $[20, 25]^\circ\text{C}$) were evaluated using a piecewise penalty function bounded within $[0, 1]$.  
> 
> Furthermore, for attributes with pronounced skewness ($S > 1.0$), an optimal Box-Cox transformation ($\hat{\lambda} = [Insert \lambda]$) was performed. As demonstrated in Figure 3, the normal Q-Q trajectory aligns closely with the theoretical reference line, verifying restored distributional symmetry.

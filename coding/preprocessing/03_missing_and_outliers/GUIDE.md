# 03 - 缺失与异常精细治理深度指南 (GUIDE)

> 本文档专为数学建模竞赛（MCM/ICM）队员编写，重点解答：**为什么缩尾处理（Winsorization）是美赛论文的大杀器？PCHIP 与普通样条（Spline）在物理仿真中有何根本区别？**

---

## 目录
1. [异常值治理策略：缩尾处理 (Winsorization) vs 粗暴删除](#一-异常值治理策略缩尾处理-winsorization-vs-粗暴删除)
2. [高保真插值：为什么物理/生物建模必须选 PCHIP？](#二-高保真插值为什么物理生物建模必须选-pchip)
3. [论文第 2 节：异常处理与缺失插补英文写作模板](#三-论文第-2-节异常处理与缺失插补英文写作模板)

---

## 一、 异常值治理策略：缩尾处理 (Winsorization) vs 粗暴删除

在处理异常值时，很多新手会直接使用 `df.dropna()` 或将异常行直接剔除。**这在美赛评委眼中是极其严重的扣分项**：
1. **样本量损失（Loss of Statistical Power）**：美赛给出的样本量往往本来就有限，每删一行就会减少自由度；
2. **多变量协同被破坏**：该样本可能只是特征 $A$ 偏大，但特征 $B, C, D$ 完全正常且极具代表性。整行删掉会把正常维度的信息全部抹杀。

### 为什么缩尾处理（Winsorization）是黄金方法？
缩尾截断在数学上并不改变样本容量 $N$，而是将超出边界的极端尾部数值压平至设定的分位阈值：
$$x_i^* = \begin{cases}
Q_1 - k \cdot \text{IQR}, & \text{if } x_i < Q_1 - k \cdot \text{IQR} \\
x_i, & \text{if } Q_1 - k \cdot \text{IQR} \le x_i \le Q_3 + k \cdot \text{IQR} \\
Q_3 + k \cdot \text{IQR}, & \text{if } x_i > Q_3 + k \cdot \text{IQR}
\end{cases}$$

* **论文辩护词**：既压制了由于传感器瞬时尖峰或突发极端干扰造成的方差膨胀，又完整保留了所有观测样本的统计自由度。

---

## 二、 高保真插值：为什么物理/生物建模必须选 PCHIP？

在 B 题（连续仿真、轨迹、水动力学、生物量演化）中，经常需要对离散采样点进行连续化插值。
传统的 **三次样条插值（Cubic Spline）** 虽然二阶导数连续，但在**阶跃点或陡峭斜率变平缓的拐角处**，会产生严重的**龙格振荡与过冲（Overshoot）**。
* **物理灾难**：在平缓的零下潜深度周围，三次样条插值可能会凭空震荡出“负深度”或“负生物量”这种反常识的非物理假象。

### PCHIP（分段三次埃尔米特保形插值）的优越性：
$$f'(x_k) = \begin{cases} 0, & \text{if } d_{k-1} d_k \le 0 \\ \frac{w_1 + w_2}{\frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}}, & \text{otherwise} \end{cases}$$
* **数学性质**：PCHIP 在插值区间上**严格保持局部的单调性（Monotonicity Preserving）**。
* **物理意义**：原数据单调上升，插值曲线必定单调上升；原数据处于两点之间，插值曲线绝不会冲出边界。因此在物理量、人口、浓度等有界物理量补齐中，**PCHIP 是首选**！

---

## 三、 论文第 2 节：异常处理与缺失插补英文写作模板

> **Section 2.3 Outlier Mitigation and Interpolation Strategy**  
> Data anomalies were identified using Tukey's Interquartile Range (IQR) criterion. Rather than discarding anomalous records—which would compromise statistical degrees of freedom and induce selection bias—we executed a two-sided Tukey Winsorization ($k = 1.5$) to project extreme values onto empirical bounding thresholds. As illustrated in Figure 2, this approach effectively attenuates severe variance inflation while retaining the complete sample population ($N = [Insert N]$).  
> 
> For missing trajectory records in the physical observation sequence, Piecewise Cubic Hermite Interpolating Polynomials (PCHIP) were deployed. Unlike conventional cubic splines that frequently generate artificial overshoot oscillations in steep transition zones, PCHIP guarantees monotonicity preservation, ensuring physically realistic non-negative trajectories throughout the simulated domain.

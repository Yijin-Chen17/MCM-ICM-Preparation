# 05 - 动态时序演化与“势头”特征工程完全指南 (GUIDE)

> 本文档专为数学建模竞赛（MCM/ICM）队员编写，重点解答：**2024 C 题这类“势头（Momentum）”在数学上到底如何严格定义与量化？滑动窗口如何杜绝未来函数数据泄露？**

---

## 目录
1. [“势头 (Momentum)”在数学上的严格建模](#一-势头-momentum在数学上的严格建模)
   - [1.1 认知心理学与马尔可夫短时记忆](#11-认知心理学与马尔可夫短时记忆)
   - [1.2 指数记忆衰减算子（EWMA）推导](#12-指数记忆衰减算子ewma推导)
2. [多尺度滑动窗口与时序数据防泄露准则](#二-多尺度滑动窗口与时序数据防泄露准则)
3. [论文第 3 节：势头量化模型英文排版与公式模板](#三-论文第-3-节势头量化模型英文排版与公式模板)

---

## 一、 “势头 (Momentum)”在数学上的严格建模

在竞技体育（2024 C 题）、金融高频交易（2022 C 题）或舆情发酵中，评委反复考查的一点是：**“势头是真实存在的物理量，还是单纯的随机游走假象？”**

### 1.1 认知心理学与马尔可夫短时记忆
* **独立同分布（i.i.d.）假设的破产**：如果认为每次得分是独立事件，那么连续得分不会增加下一次得分概率。但真实人类存在“自信心增强、手感火热（Hot Hand）”以及“失误导致的心理崩溃与认知负荷增加”。
* **记忆衰减特性**：刚刚发生的一个得分，对心理的振奋程度远大于 10 分钟之前发生的一个得分。

---

### 1.2 指数记忆衰减算子（EWMA）推导
设在时间步 $t$ 时刻，选手的瞬时表现为 $Y_t \in \{-1, +1\}$。
定义即时动态势头 $M_t$ 为表现序列的离散时间指数衰减加权累和：
$$M_t = \sum_{k=0}^{t-1} w_k Y_{t-k} = \alpha Y_t + (1 - \alpha) M_{t-1}$$
其中 $\alpha \in (0, 1]$ 为**记忆衰减更新率（Memory Update Coefficient）**。

展开为显式级数形式：
$$M_t = \alpha \left[ Y_t + (1-\alpha)Y_{t-1} + (1-\alpha)^2 Y_{t-2} + \dots + (1-\alpha)^t Y_0 \right]$$

* **物理含义**：
  * 权重的几何级数收敛 $\sum_{k=0}^\infty \alpha(1-\alpha)^k = 1$；
  * 越靠近当前时刻 $t$，权重越大；历史事件的影响以速率 $(1-\alpha)$ 呈指数形式衰减（Half-life 半衰期为 $h = \frac{\ln 0.5}{\ln(1-\alpha)}$）。

---

## 二、 多尺度滑动窗口与时序数据防泄露准则

在将微观事件流转为机器学习特征时，许多队伍被扣分的隐蔽原因是**未来信息泄露（Lookahead Bias）**：
* **致命错误**：在时刻 $t$ 的滑动窗口包含了从 $t - w/2$ 到 $t + w/2$ 的居中窗口（`center=True`），提前“偷窥”了未来的比赛走向；
* **正确原则（本工具箱标准实现）**：必须使用**因果窗口（Causal / Backward-Looking Window）**：
  $$\bar{X}_t^{(w)} = \frac{1}{w} \sum_{i=0}^{w-1} X_{t-i}$$
  所有特征在时刻 $t$ 时刻，仅由 $\{t, t-1, t-2, \dots\}$ 决定，确保模型推断在真实时间轴上严格因果闭环。

---

## 三、 论文第 3 节：势头量化模型英文排版与公式模板

> **Section 3.1 Mathematical Formulation of Dynamic Match Momentum**  
> To operationalize the latent psychological and performance state—commonly conceptualized as "momentum"—we developed a discrete-time exponential decay state-space operator. Given instantaneous point outcome $y_t \in \{-1, +1\}$ at event index $t$, the dynamic momentum metric $M_t$ evolves according to:  
> $$M_t = \alpha \cdot y_t + (1 - \alpha) \cdot M_{t-1}, \quad M_0 = 0$$  
> where $\alpha = 0.3$ serves as the memory updating parameter, corresponding to an empirical half-life of approximately two scoring sequences.  
> 
> As visualized in Figure 4, positive deviations ($M_t > 0$, teal shaded zone) signify dominant control periods, whereas cross-zero transitions represent tactical regime shifts. Furthermore, running streak lengths $L_t$ and multi-horizon rolling volatility ($\text{Std}^{(5)}$) were integrated to capture system instability preceding turning points.

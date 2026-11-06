# 基准模型库 (Baseline Models)

本目录包含用于数学建模竞赛的经典基准模型脚手架。基准模型用于在引入复杂机器学习或深度学习算法之前，建立一个具有统计解释力与可度量性能基线的参照系统。

---

## 模型索引

* [**01 - 线性回归基准模型 (1-linear_regression)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/baseline/1-linear_regression/README.md)
  * **算法类型**：普通最小二乘回归（Ordinary Least Squares, OLS）
  * **核心功能**：特征标准化、Box-Cox 目标变换、二次项/交互项扩展、LINE 原则统计假设检验（VIF、Breusch-Pagan、Durbin-Watson、Jarque-Bera）、诊断图表绘制及 LaTeX 格式三线表输出。
  * **适用场景**：连续型因变量预测、变量边际效应与机理分析、可解释政策建议推导。

---

## 依赖配置

```bash
pip install -r requirements.txt
```
核心依赖：`numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`。

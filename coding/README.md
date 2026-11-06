# Coding / 算法与代码资料库

本目录存储用于美国大学生数学建模竞赛（MCM/ICM）的算法实现、数据预处理工具以及基准模型。所有代码均遵循模块化与工程化设计，提供防御性参数校验与学术图表/LaTeX 表格导出功能。

---

## 目录结构与模块索引

### 1. 基准模型库 ([baseline](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/baseline/README.md))
用于在正式建模前快速构建参照基准（Baseline），评估后续复杂模型的提升幅度，并提供可解释的统计推断结果。

* [**01 - 线性回归基准模型 (1-linear_regression)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/baseline/1-linear_regression/README.md)：实现普通最小二乘法（OLS）回归建模、LINE 原则统计假设检验（共线性 VIF、异方差、正态性、自相关）、诊断图表绘制与 LaTeX 回归三线表导出。

---

### 2. 数据预处理工具箱 ([preprocessing](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/README.md))
提供针对结构化表格、离散事件流、时序数据与空间连续数据的原子化预处理函数库。各子模块功能如下：

* [**01 - 数据体检与描述性统计 (01_eda_and_audit)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/01_eda_and_audit/README.md)：计算数据集的缺失率、分布分位数、偏度、峰度与相关系数矩阵，支持一键导出论文 Table 1 描述性统计三线表。
* [**02 - 字符串清洗与类别数值化 (02_strings_and_categorical)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/02_strings_and_categorical/README.md)：提供含单位/货币符号的字符串转数值、二元变量映射、有序分类编码、低频截断独热编码（One-Hot）与交叉验证平滑目标编码（Target Encoding）。
* [**03 - 缺失与异常精细治理 (03_missing_and_outliers)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/03_missing_and_outliers/README.md)：基于 Tukey IQR 与 Z-score 的异常值掩码检测、保持样本量的缩尾处理（Winsorization）、分段三次埃尔米特保形插值（PCHIP）与多维 KNN 插补。
* [**04 - 分布矫正与指标无量纲化 (04_transforms_and_scaling)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/04_transforms_and_scaling/README.md)：Box-Cox 与 Yeo-Johnson 幂变换正态化修正、效益型/成本型/区间最优型指标正向化映射、TOPSIS 向量规范化与熵权法防零处理。
* [**05 - 时序演化与动量特征工程 (05_timeseries_and_momentum)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/05_timeseries_and_momentum/README.md)：因果滑动窗口多尺度统计量计算、指数加权移动平均（EWMA）动量建模、连续状态（Streak）统计与无未来信息泄露的滞后差分。
* [**06 - 空间几何网格化与信号平滑 (06_spatial_and_smoothing)**](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/06_spatial_and_smoothing/README.md)：Savitzky-Golay 局部多项式保形滤波与解析导数求取、2D/3D 空间搜索区域栅格化剖分与经纬度球面大圆距离（Haversine）矩阵计算。

---

## 环境配置与依赖

本目录分为两套独立的依赖清单：
* **数据预处理依赖**：[`coding/preprocessing/requirements.txt`](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/preprocessing/requirements.txt)（仅依赖 `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`）
* **基准模型依赖**：[`coding/baseline/requirements.txt`](file:///Users/ricardo/Codes/MCM-ICM-Preparation/coding/baseline/requirements.txt)（增加 `statsmodels`, `scikit-learn`）

安装命令：
```bash
pip install -r coding/preprocessing/requirements.txt
pip install -r coding/baseline/requirements.txt
```

---

## 模块文档规范

每个功能子目录均包含以下三个标准文件：
1. `*.py`：核心算法或工具函数的 Python 实现，提供详细类型标注与防御性异常处理；
2. `README.md`：函数签名速查、输入输出说明与调用示例；
3. `GUIDE.md`：数学公式推导、参数选型依据与论文中对应章节的英文撰写模板。

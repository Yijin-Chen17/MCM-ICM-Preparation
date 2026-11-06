"""
MCM/ICM Baseline Series: 01 - Linear Regression
美国大学生数学建模竞赛基准模型模版：01 - 线性回归 (OLS)

功能概述：
1. 灵活的数据集管理：
   - 默认 split_data=False 进行全量数据建模与诊断
   - 支持 split_data=True 内置 8:2 切分
   - 支持传入外部 test_df 独立测试集
2. 多样化特征缩放机制 (feature_scaling)：
   - 'zscore': Z-score 标准化 (均值 0，方差 1，默认)
   - 'minmax': Min-Max 归一化 (压缩至 [0, 1])
   - 'robust': 鲁棒缩放 (基于中位数与四分位距 IQR，免疫离群值)
   - 'centering': 仅均值中心化 (消除截距共线性，配合高阶项)
   - None: 保持原特征尺度
3. 丰富的因变量分布修正 (target_transform)：
   - None: 原始目标
   - 'log': 对数变换 ln(Y) (缓解右偏与异方差，构建弹性模型)
   - 'boxcox': Box-Cox 幂变换 (最大似然自适应寻优恢复正态性)
   - 'sqrt': 平方根变换 sqrt(Y) (计数/泊松数据方差齐性)
   *注：评估指标与诊断出图时自动逆变换还原为原始物理单位，确保模型横向对比公平！*
4. 高级特征结构拓展：
   - log_features: 指定特定右偏长尾特征自动取对数
   - add_interactions: 自动引入中心化交互项 (X_i * X_j)
   - add_polynomial: 自动引入中心化二次项 (X_i^2，捕捉倒 U 形拐点)
5. 严谨的数理统计假设检验 (LINE 原则)：
   - 多重共线性检验 (VIF 排序表)
   - 残差自相关检验 (Durbin-Watson 统计量)
   - 异方差检验 (Breusch-Pagan 检验 p 值)
   - 残差正态性检验 (Jarque-Bera 检验 p 值)
6. 美赛级 4 合 1 高清学术出图 (300 DPI)
7. 一键生成标准学术三线表 (LaTeX Booktabs 代码，带显著性星号)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import inv_boxcox
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MCMRegressionBaseline:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list = None,
        # --- 数据划分设置 ---
        split_data: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        test_df: pd.DataFrame = None,
        # --- 变换与缩放选项 (可根据赛题手动切换) ---
        feature_scaling: str = 'zscore',    # 'zscore', 'minmax', 'robust', 'centering', None
        target_transform: str = None,       # None, 'log', 'boxcox', 'sqrt'
        log_features: list = None,          # 需要取对数的特征列名列表，例如 ['GDP', 'Population']
        add_polynomial: bool = False,       # 是否自动增加二次项 (X^2，经中心化处理防共线性)
        add_interactions: bool = False      # 是否自动增加特征交互项 (X_i * X_j，经中心化处理)
    ):
        """
        初始化高灵活性回归基准模型流水线
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols is not None else [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
        ]
        self.split_data = split_data
        self.test_size = test_size
        self.random_state = random_state
        self.test_df = test_df.copy() if test_df is not None else None
        
        # 变换参数
        self.feature_scaling = feature_scaling
        self.target_transform = target_transform
        self.log_features = log_features if log_features is not None else []
        self.add_polynomial = add_polynomial
        self.add_interactions = add_interactions
        
        # 变换状态存储
        self.boxcox_lambda = None
        self.scaler = None
        self.final_feature_names = []
        
        # 运行时属性
        self.has_test = False
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.y_train_orig, self.y_test_orig = None, None  # 记录原始未变换的真实 Y，用于计算真实物理 RMSE
        self.model = None          # 核心 OLS 模型 (statsmodels)
        self.std_model = None      # 标准化 OLS 模型 (用于衡量特征重要性)
        self.metrics = {}          # 评估指标
        self.vif_df = None         # VIF 多重共线性表
        
    def _transform_target(self, y_series: pd.Series, is_train: bool = True):
        """对目标因变量 Y 执行数学变换"""
        if self.target_transform is None:
            return y_series.values
        elif self.target_transform == 'log':
            # 若含有非正数，自动使用 log1p (ln(y + 1))
            if (y_series <= 0).any():
                min_val = y_series.min()
                shift = abs(min_val) + 1.0 if min_val <= 0 else 0
                return np.log(y_series.values + shift)
            return np.log(y_series.values)
        elif self.target_transform == 'boxcox':
            y_arr = y_series.values
            shift = abs(y_arr.min()) + 1.0 if (y_arr <= 0).any() else 0
            y_shifted = y_arr + shift
            if is_train:
                y_trans, lmbda = stats.boxcox(y_shifted)
                self.boxcox_lambda = (lmbda, shift)
                return y_trans
            else:
                lmbda, s = self.boxcox_lambda
                return stats.boxcox(y_arr + s, lmbda=lmbda)
        elif self.target_transform == 'sqrt':
            y_arr = y_series.values
            shift = abs(y_arr.min()) if (y_arr < 0).any() else 0
            return np.sqrt(y_arr + shift)
        else:
            raise ValueError(f"未知的目标变量变换类型: {self.target_transform}")

    def _inverse_transform_target(self, y_pred_trans):
        """将预测值逆变换还原为原始物理量纲"""
        if self.target_transform is None:
            return y_pred_trans
        elif self.target_transform == 'log':
            return np.exp(y_pred_trans)
        elif self.target_transform == 'boxcox':
            lmbda, shift = self.boxcox_lambda
            return inv_boxcox(y_pred_trans, lmbda) - shift
        elif self.target_transform == 'sqrt':
            return np.square(y_pred_trans)
        return y_pred_trans

    def _build_feature_matrix(self, X_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """根据配置自动生成特征矩阵（包含对数特征、中心化交互项与二次项）"""
        X_mat = X_df.copy()
        
        # 1. 对指定长尾特征取对数 (Log Features)
        for col in self.log_features:
            if col in X_mat.columns:
                shift = abs(X_mat[col].min()) + 1.0 if (X_mat[col] <= 0).any() else 0
                X_mat[f"log_{col}"] = np.log(X_mat[col] + shift)
                X_mat.drop(columns=[col], inplace=True)
                
        base_cols = list(X_mat.columns)
        
        # 2. 中心化辅助（构造高阶项前先中心化，极大缓解多重共线性）
        X_centered = X_mat - X_mat.mean()
        
        # 3. 增加二次多项式项 (Polynomial X^2)
        if self.add_polynomial:
            for col in base_cols:
                X_mat[f"{col}_sq"] = X_centered[col] ** 2
                
        # 4. 增加两两交互项 (Interactions X_i * X_j)
        if self.add_interactions:
            for i in range(len(base_cols)):
                for j in range(i + 1, len(base_cols)):
                    c1, c2 = base_cols[i], base_cols[j]
                    X_mat[f"{c1}x{c2}"] = X_centered[c1] * X_centered[c2]
                    
        if is_train:
            self.final_feature_names = list(X_mat.columns)
        return X_mat

    def fit(self):
        """执行模型训练、统计检验与指标计算全流程"""
        # 1. 确定原始 X 与 y
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        # 2. 数据切分逻辑
        if self.test_df is not None:
            self.has_test = True
            X_train_raw, y_train_raw = X, y
            X_test_raw = self.test_df[self.feature_cols]
            y_test_raw = self.test_df[self.target_col]
        elif self.split_data:
            self.has_test = True
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        else:
            self.has_test = False
            X_train_raw, y_train_raw = X, y
            X_test_raw, y_test_raw = None, None
            
        self.y_train_orig = y_train_raw.values
        if self.has_test:
            self.y_test_orig = y_test_raw.values

        # 3. 特征拓展与构建 (Log, Polynomial, Interactions)
        self.X_train = self._build_feature_matrix(X_train_raw, is_train=True)
        if self.has_test:
            self.X_test = self._build_feature_matrix(X_test_raw, is_train=False)

        # 4. 目标变量变换 (Target Transform)
        self.y_train = self._transform_target(y_train_raw, is_train=True)
        if self.has_test:
            self.y_test = self._transform_target(y_test_raw, is_train=False)

        # 5. 拟合核心模型 (可保留原始或指定缩放)
        X_train_const = sm.add_constant(self.X_train)
        self.model = sm.OLS(self.y_train, X_train_const).fit()

        # 6. 拟合标准化模型 (用于横向对比特征重要性排行榜)
        # 支持 'zscore', 'minmax', 'robust', 'centering'
        if self.feature_scaling == 'minmax':
            scaler_x = MinMaxScaler()
        elif self.feature_scaling == 'robust':
            scaler_x = RobustScaler()
        elif self.feature_scaling == 'centering':
            class CenteringScaler:
                def fit_transform(self, arr): return arr - arr.mean(axis=0)
            scaler_x = CenteringScaler()
        else:
            scaler_x = StandardScaler()  # 默认 z-score

        scaler_y = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(self.X_train)
        y_train_std = scaler_y.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.std_model = sm.OLS(y_train_std, sm.add_constant(X_train_scaled)).fit()

        # 7. 计算评估指标 (包含模型尺度指标和原始物理量纲还原指标)
        train_pred_trans = self.model.predict(X_train_const)
        train_pred_orig = self._inverse_transform_target(train_pred_trans)
        rmse_train_orig = np.sqrt(mean_squared_error(self.y_train_orig, train_pred_orig))
        mae_train_orig = mean_absolute_error(self.y_train_orig, train_pred_orig)

        if self.has_test:
            X_test_const = sm.add_constant(self.X_test)
            test_pred_trans = self.model.predict(X_test_const)
            test_pred_orig = self._inverse_transform_target(test_pred_trans)
            
            self.metrics = {
                "Model R2 (Train)": self.model.rsquared,
                "Model Adj. R2": self.model.rsquared_adj,
                "Test R2 (Orig Scale)": r2_score(self.y_test_orig, test_pred_orig),
                "Test RMSE (Orig Scale)": np.sqrt(mean_squared_error(self.y_test_orig, test_pred_orig)),
                "Test MAE (Orig Scale)": mean_absolute_error(self.y_test_orig, test_pred_orig),
                "F-statistic": self.model.fvalue,
                "F p-value": self.model.f_pvalue
            }
        else:
            self.metrics = {
                "R2": self.model.rsquared,
                "Adj. R2": self.model.rsquared_adj,
                "RMSE (Orig Scale)": rmse_train_orig,
                "MAE (Orig Scale)": mae_train_orig,
                "F-statistic": self.model.fvalue,
                "F p-value": self.model.f_pvalue
            }

        # 8. 计算多重共线性 (VIF)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.final_feature_names
        vif_data["VIF"] = [
            variance_inflation_factor(self.X_train.values, i)
            for i in range(len(self.final_feature_names))
        ]
        self.vif_df = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
        return self

    def diagnose(self) -> dict:
        """执行统计学假设检验（LINE 原则与统计前提验证）"""
        residuals = self.model.resid
        dw = durbin_watson(residuals)
        bp_test = het_breuschpagan(residuals, sm.add_constant(self.X_train))
        bp_pvalue = bp_test[1]
        jb_pvalue = stats.jarque_bera(residuals)[1]
        
        return {
            "Durbin-Watson (No Autocorr ~ 2.0)": round(dw, 4),
            "Breusch-Pagan p-val (Homoscedasticity > 0.05)": round(bp_pvalue, 4),
            "Jarque-Bera p-val (Normality > 0.05)": round(jb_pvalue, 4),
            "Max VIF (Ideal < 10)": round(self.vif_df["VIF"].max(), 4)
        }

    def plot_actual_vs_predicted(self, save_path: str = None, ax=None):
        """
        图 (1): 真实值 vs 预测/拟合值散点图 (已还原真实物理尺度，含 y=x 完美拟合参考线)
        """
        is_standalone = ax is None
        if is_standalone:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
            
        if self.has_test:
            X_test_const = sm.add_constant(self.X_test)
            pred_orig = self._inverse_transform_target(self.model.predict(X_test_const))
            y_actual = self.y_test_orig
            ax.set_title("Test Set: Actual vs. Predicted (Original Scale)", fontsize=11, fontweight='bold')
        else:
            pred_orig = self._inverse_transform_target(self.model.fittedvalues)
            y_actual = self.y_train_orig
            ax.set_title("Actual vs. Fitted Values (Full Data, Original Scale)", fontsize=11, fontweight='bold')
            
        ax.scatter(y_actual, pred_orig, alpha=0.7, edgecolors='k', color='#2b5c8f')
        min_v = min(y_actual.min(), pred_orig.min())
        max_v = max(y_actual.max(), pred_orig.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label="Perfect Fit (y = x)")
        ax.set_xlabel("Actual Values", fontsize=10)
        ax.set_ylabel("Predicted Values", fontsize=10)
        ax.legend(frameon=True)
        
        if is_standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"[✓] 真实值 vs 预测值图已保存至: {save_path}")
            plt.show()
        return ax

    def plot_residuals_vs_fitted(self, save_path: str = None, ax=None):
        """
        图 (2): 残差 vs 拟合值散点图 (检验同方差性与未建模的非线性趋势)
        """
        is_standalone = ax is None
        if is_standalone:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
            
        residuals = self.model.resid
        fitted = self.model.fittedvalues
        
        ax.scatter(fitted, residuals, alpha=0.7, edgecolors='k', color='#4682B4')
        ax.axhline(0, color='red', linestyle='--', lw=2)
        ax.set_title("Residuals vs. Fitted (Homoscedasticity Check)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Fitted Values", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        
        if is_standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"[✓] 残差 vs 拟合值图已保存至: {save_path}")
            plt.show()
        return ax

    def plot_qq(self, save_path: str = None, ax=None):
        """
        图 (3): 正态 Q-Q 图 (检验残差分布的正态性假设)
        """
        is_standalone = ax is None
        if is_standalone:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
            
        residuals = self.model.resid
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor('#2E8B57')
        ax.get_lines()[0].set_markeredgecolor('k')
        ax.set_title("Normal Q-Q Plot (Residual Normality)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Theoretical Quantiles", fontsize=10)
        ax.set_ylabel("Sample Quantiles", fontsize=10)
        
        if is_standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"[✓] 正态 Q-Q 图已保存至: {save_path}")
            plt.show()
        return ax

    def plot_feature_importance(self, save_path: str = None, ax=None):
        """
        图 (4): 标准化特征重要性水平条形图 (横向对比驱动因素正负向强弱)
        """
        is_standalone = ax is None
        if is_standalone:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
            
        std_coefs = self.std_model.params[1:]
        sorted_indices = np.argsort(np.abs(std_coefs))
        y_pos = np.arange(len(std_coefs))
        
        colors = ['#d9534f' if c < 0 else '#5cb85c' for c in std_coefs.iloc[sorted_indices]]
        ax.barh(y_pos, std_coefs.iloc[sorted_indices], color=colors, edgecolor='black', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.final_feature_names[i] for i in sorted_indices], fontsize=9)
        scaler_name = self.feature_scaling.upper() if self.feature_scaling else "STANDARDIZED"
        ax.set_title(f"Feature Importance ({scaler_name} Beta)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Standardized Coefficient Magnitude", fontsize=10)
        
        if is_standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"[✓] 特征重要性条形图已保存至: {save_path}")
            plt.show()
        return ax

    def plot_diagnostics(self, save_dir: str = None, combine_4in1: bool = False, show: bool = True):
        """
        统一诊断出图控制器：
        1. 默认分别生成 4 张独立的诊断高清图 (可指定 save_dir 自动批量保存独立单图)
        2. 可选 combine_4in1=True：额外或单独合并生成一张 2x2 的学术四合一大图
        :param save_dir: 保存图片的文件夹路径 (如 'figures/' 或 './')
        :param combine_4in1: 是否额外生成并保存 4 合 1 大图 (默认 False 关闭)
        :param show: 是否在交互窗口中显示图形 (默认 True)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1. 分别生成 4 张独立的高清图
        path1 = os.path.join(save_dir, "1_actual_vs_predicted.png") if save_dir else None
        path2 = os.path.join(save_dir, "2_residuals_vs_fitted.png") if save_dir else None
        path3 = os.path.join(save_dir, "3_normal_qq.png") if save_dir else None
        path4 = os.path.join(save_dir, "4_feature_importance.png") if save_dir else None

        self.plot_actual_vs_predicted(save_path=path1)
        self.plot_residuals_vs_fitted(save_path=path2)
        self.plot_qq(save_path=path3)
        self.plot_feature_importance(save_path=path4)

        # 2. 若用户指定 combine_4in1=True，则额外生成并保存一张 4 合 1 大图
        if combine_4in1:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
            
            self.plot_actual_vs_predicted(ax=axes[0, 0])
            self.plot_residuals_vs_fitted(ax=axes[0, 1])
            self.plot_qq(ax=axes[1, 0])
            self.plot_feature_importance(ax=axes[1, 1])
            
            axes[0, 0].set_title("(a) " + axes[0, 0].get_title(), fontsize=11, fontweight='bold')
            axes[0, 1].set_title("(b) " + axes[0, 1].get_title(), fontsize=11, fontweight='bold')
            axes[1, 0].set_title("(c) " + axes[1, 0].get_title(), fontsize=11, fontweight='bold')
            axes[1, 1].set_title("(d) " + axes[1, 1].get_title(), fontsize=11, fontweight='bold')

            plt.tight_layout()
            if save_dir:
                combo_path = os.path.join(save_dir, "diagnostic_4in1.png")
                plt.savefig(combo_path, bbox_inches='tight')
                print(f"[✓] 四合一诊断大图已保存至: {combo_path}")
            if show:
                plt.show()
            else:
                plt.close(fig)

    def to_latex_table(self) -> str:
        """生成学术标准三线表 (LaTeX Booktabs 代码)"""
        params = self.model.params
        bse = self.model.bse
        tvalues = self.model.tvalues
        pvalues = self.model.pvalues
        
        def star(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            return ""

        latex = "\\begin{table}[htbp]\n\\centering\n"
        caption_info = f"OLS Regression (Target: {self.target_transform or 'Raw'}, Scaling: {self.feature_scaling or 'None'})"
        latex += f"\\caption{{{caption_info}}}\n"
        latex += "\\label{tab:ols_baseline}\n"
        latex += "\\begin{tabular}{lcccc}\n\\toprule\n"
        latex += "\\textbf{Variable} & \\textbf{Coefficient} & \\textbf{Std. Error} & \\textbf{$t$-statistic} & \\textbf{$p$-value} \\\\\n\\midrule\n"
        
        for name in params.index:
            p_val = pvalues[name]
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            var_name = "\\textit{const}" if name == "const" else name.replace("_", "\\_")
            latex += f"{var_name} & {params[name]:.4f}{star(p_val)} & {bse[name]:.4f} & {tvalues[name]:.3f} & {p_str} \\\\\n"
            
        latex += "\\midrule\n"
        if self.has_test:
            latex += f"Train $R^2$ & {self.metrics['Model R2 (Train)']:.4f} & Adj. $R^2$ & {self.metrics['Model Adj. R2']:.4f} \\\\\n"
            latex += f"Test RMSE & {self.metrics['Test RMSE (Orig Scale)']:.4f} & Test $R^2$ & {self.metrics['Test R2 (Orig Scale)']:.4f} \\\\\n"
        else:
            latex += f"$R^2$ & {self.metrics['R2']:.4f} & Adj. $R^2$ & {self.metrics['Adj. R2']:.4f} \\\\\n"
            latex += f"RMSE & {self.metrics['RMSE (Orig Scale)']:.4f} & MAE & {self.metrics['MAE (Orig Scale)']:.4f} \\\\\n"
        latex += "\\bottomrule\n"
        latex += "\\multicolumn{5}{l}{\\footnotesize * $p<0.05$, ** $p<0.01$, *** $p<0.001$} \\\\\n"
        latex += "\\end{tabular}\n\\end{table}"
        return latex

    def summary(self):
        """终端打印详细摘要报告"""
        print("=" * 65)
        print("                MCM/ICM REGRESSION BASELINE REPORT                ")
        print("=" * 65)
        print(f"Target Variable (Y): {self.target_col} | Transform: {self.target_transform or 'None'}")
        if self.boxcox_lambda:
            print(f"Box-Cox Lambda: {self.boxcox_lambda[0]:.4f} (Shift={self.boxcox_lambda[1]:.2f})")
        print(f"Features: {len(self.final_feature_names)} features -> {self.final_feature_names}")
        print(f"Feature Scaling: {self.feature_scaling} | Log Features: {self.log_features or 'None'}")
        print(f"Structure Expansion: Polynomial={self.add_polynomial}, Interactions={self.add_interactions}")
        split_status = f"Split Enabled (test_size={self.test_size})" if self.split_data else (
            "External Test Set Provided" if self.test_df is not None else "Disabled (Full Dataset Modeling)"
        )
        print(f"Data Split Mode: {split_status}")
        print("-" * 65)
        print("[1] Evaluation Metrics:")
        for k, v in self.metrics.items():
            print(f"    - {k:<25}: {v:.4f}")
        print("-" * 65)
        print("[2] Statistical Diagnostics (LINE Assumptions):")
        for k, v in self.diagnose().items():
            print(f"    - {k:<45}: {v}")
        print("-" * 65)
        print("[3] Multicollinearity Check (VIF Table):")
        print(self.vif_df.to_string(index=False))
        print("=" * 65)


# ==============================================================================
# 示例演示：比赛中如何根据赛题特征灵活切换配置
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 300
    
    # 模拟长尾分布特征 (如 GDP、人口、奖牌)
    gdp = np.exp(np.random.normal(3.0, 0.8, n_samples))          # 偏态对数正态
    population = np.exp(np.random.normal(2.5, 0.6, n_samples))   # 偏态对数正态
    athletes = 0.5 * np.log(gdp) + 0.3 * np.log(population) + np.random.normal(0, 0.2, n_samples)
    
    # 真实关系（乘积效应 + 对数响应）
    true_score = 5.0 * (gdp ** 0.4) * (athletes ** 0.6) * np.exp(np.random.normal(0, 0.15, n_samples))
    
    demo_df = pd.DataFrame({
        "GDP": gdp,
        "Population": population,
        "Athletes_Count": athletes,
        "Medal_Score": true_score
    })
    
    print("[INFO] 正在演示灵活配置的 Baseline 拟合...")
    
    # 【实战调优演示】：
    # 1. 对长尾因变量 Medal_Score 取 log (构建对数模型缓解偏态)
    # 2. 对长尾自变量 ['GDP', 'Population'] 自动取对数
    # 3. 开启中心化交互项 (add_interactions=True) 捕捉协同效应
    # 4. 特征重要性排序选用 'robust' 鲁棒缩放
    baseline = MCMRegressionBaseline(
        demo_df, 
        target_col="Medal_Score",
        target_transform='log',              # 支持: None, 'log', 'boxcox', 'sqrt'
        log_features=['GDP', 'Population'],  # 指定右偏特征自动取对数
        feature_scaling='robust',            # 支持: 'zscore', 'minmax', 'robust', 'centering'
        add_interactions=True,               # 自动增加中心化交互项
        split_data=False                     # 保持默认关闭切分
    ).fit()
    
    # 终端输出全面体检与评估报告
    baseline.summary()
    
    # 输出 LaTeX 代码
    print("\n[LaTeX 三线表代码]:\n")
    print(baseline.to_latex_table())
    
    # 绘制与保存诊断图（演示不同出图方式）：
    # 方式 1: 单独导出某一张图 (例如只想要特征重要性图放入论文第4节)
    # baseline.plot_feature_importance(save_path="feature_importance.png")
    
    # 方式 2: 一键批量导出 4 张独立图表，并可选额外生成一张 4 合 1 大图
    # baseline.plot_diagnostics(save_dir="output_plots/", combine_4in1=True)

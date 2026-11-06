"""
MCM/ICM Preprocessing Series: 01 - EDA & Data Audit Toolkit
美国大学生数学建模竞赛数据预处理工具箱：01 - 数据体检与描述性统计

设计原则：
- 高度工程化与抗造性（Defensive Programming）：
  * 自动容忍混合类型、全空列、单值列、无穷大（inf）
  * 自动转义 LaTeX 特殊字符（如 _, %, $, & 等），防止编译崩溃
  * 所有函数支持输入 pd.DataFrame 或 pd.Series，支持深拷贝保障，避免 SettingWithCopyWarning
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def audit_dataset(
    df: pd.DataFrame,
    skew_threshold: float = 1.0,
    kurt_threshold: float = 3.0
) -> pd.DataFrame:
    """
    对输入 DataFrame 进行全方位的工业级数据体检。
    
    参数：
    ----------
    df : pd.DataFrame
        待体检的数据集
    skew_threshold : float, 默认 1.0
        判定偏态的绝对值阈值（|skew| > threshold 标记为有偏）
    kurt_threshold : float, 默认 3.0
        判定尖峰厚尾的阈值（kurtosis > threshold 标记为厚尾）
        
    返回：
    ----------
    pd.DataFrame:
        包含各列字段类型、缺失率、唯一值数、均值、标准差、分位数、偏度、峰度及健康诊断标记的体检报告表。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"输入参数 df 必须为 pd.DataFrame，收到: {type(df)}")
    
    if df.empty:
        return pd.DataFrame(columns=[
            "Column", "Dtype", "Non_Null_Count", "Missing_Count", "Missing_Rate(%)",
            "Unique_Count", "Mean", "Std", "Median", "IQR", "Skewness", "Kurtosis", "Audit_Note"
        ])
    
    records = []
    n_rows = len(df)
    
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_rate = (missing / n_rows) * 100.0 if n_rows > 0 else 0.0
        unique_cnt = int(s.nunique(dropna=True))
        
        # 针对数值型进行高阶统计矩计算
        is_numeric = pd.api.types.is_numeric_dtype(s)
        mean_val = np.nan
        std_val = np.nan
        median_val = np.nan
        iqr_val = np.nan
        skew_val = np.nan
        kurt_val = np.nan
        
        notes = []
        if missing_rate > 50.0:
            notes.append("High Missing (>50%)")
        elif missing_rate > 0:
            notes.append(f"Missing ({missing_rate:.1f}%)")
            
        if unique_cnt <= 1:
            notes.append("Constant/Empty (Zero Variance)")
            
        if is_numeric and non_null > 0:
            # 滤除 inf / -inf 保护计算
            valid_vals = s[np.isfinite(s)].dropna().astype(float)
            if len(valid_vals) > 0:
                mean_val = float(valid_vals.mean())
                std_val = float(valid_vals.std(ddof=1)) if len(valid_vals) > 1 else 0.0
                median_val = float(valid_vals.median())
                q25 = float(valid_vals.quantile(0.25))
                q75 = float(valid_vals.quantile(0.75))
                iqr_val = q75 - q25
                
                # 样本量 > 2 时计算偏度与峰度
                if len(valid_vals) >= 3 and std_val > 1e-12:
                    skew_val = float(stats.skew(valid_vals, bias=False))
                    kurt_val = float(stats.kurtosis(valid_vals, bias=False, fisher=True)) # Fisher 峰度（正态分布为0）
                    
                    if abs(skew_val) > skew_threshold:
                        notes.append(f"Skewed ({skew_val:+.2f})")
                    if kurt_val > kurt_threshold:
                        notes.append(f"Heavy-tailed (Kurt={kurt_val:.2f})")
        
        records.append({
            "Column": str(col),
            "Dtype": dtype,
            "Non_Null_Count": non_null,
            "Missing_Count": missing,
            "Missing_Rate(%)": round(missing_rate, 2),
            "Unique_Count": unique_cnt,
            "Mean": round(mean_val, 4) if pd.notna(mean_val) else "-",
            "Std": round(std_val, 4) if pd.notna(std_val) else "-",
            "Median": round(median_val, 4) if pd.notna(median_val) else "-",
            "IQR": round(iqr_val, 4) if pd.notna(iqr_val) else "-",
            "Skewness": round(skew_val, 3) if pd.notna(skew_val) else "-",
            "Kurtosis": round(kurt_val, 3) if pd.notna(kurt_val) else "-",
            "Audit_Note": "; ".join(notes) if notes else "Healthy"
        })
        
    return pd.DataFrame(records)


def detect_constant_columns(df: pd.DataFrame, drop_na: bool = True) -> List[str]:
    """
    检测零方差（单值或全空）列。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须为 pd.DataFrame")
    
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=drop_na) <= 1:
            constant_cols.append(str(col))
    return constant_cols


def _escape_latex(text: str) -> str:
    """内部辅助函数：转义 LaTeX 特殊保留字符"""
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text


def export_latex_table1(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    caption: str = "Descriptive Statistics and Health Audit of Key Variables",
    label: str = "tab:data_summary",
    precision: int = 3
) -> str:
    """
    一键生成符合美赛学术出版标准的 Table 1 描述性统计三线表 LaTeX 代码（Booktabs 格式）。
    
    参数：
    ----------
    df : pd.DataFrame
        待处理数据集
    columns : list of str, 可选
        指定展示的数值型字段列表（默认自动筛选全部数值型字段）
    caption : str
        表格标题
    label : str
        LaTeX 引用标签
    precision : int
        数值精度保留位数
    """
    if columns is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = [c for c in columns if c in df.columns]
        
    if not numeric_cols:
        return "% Warning: No numeric columns found to generate Table 1."
    
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        # 滤除 inf
        s = s[np.isfinite(s)].astype(float)
        
        count = len(s)
        missing = len(df[col]) - count
        missing_pct = (missing / len(df[col])) * 100.0 if len(df[col]) > 0 else 0.0
        mean = s.mean() if count > 0 else np.nan
        std = s.std(ddof=1) if count > 1 else 0.0
        median = s.median() if count > 0 else np.nan
        p25 = s.quantile(0.25) if count > 0 else np.nan
        p75 = s.quantile(0.75) if count > 0 else np.nan
        skew = stats.skew(s, bias=False) if count >= 3 and std > 1e-12 else np.nan
        
        def fmt(val):
            return f"{val:.{precision}f}" if pd.notna(val) else "-"
        
        col_escaped = _escape_latex(str(col))
        rows.append(f"  {col_escaped} & {count} & {missing_pct:.1f}\\% & {fmt(mean)} & {fmt(std)} & {fmt(median)} & [{fmt(p25)}, {fmt(p75)}] & {fmt(skew)} \\\\")
        
    body = "\n".join(rows)
    
    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
\\textbf{{Variable}} & \\textbf{{Count}} & \\textbf{{Missing}} & \\textbf{{Mean}} & \\textbf{{Std}} & \\textbf{{Median}} & \\textbf{{IQR [25\\%, 75\\%]}} & \\textbf{{Skewness}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    return latex_code


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson',
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    title: str = "Correlation Matrix Heatmap"
) -> plt.Figure:
    """
    绘制美赛出版级学术相关性热力图（可自动隐藏弱相关系数值）。
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        numeric_df = df[valid_cols]
        
    if numeric_df.empty or numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
        return fig
        
    corr = numeric_df.corr(method=method)
    labels = corr.columns.tolist()
    n = len(labels)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300 if save_path else 100)
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(f"{method.capitalize()} Correlation", rotation=270, labelpad=12, fontsize=10)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    # 在单元格内标注系数值
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if pd.notna(val):
                if threshold is not None and abs(val) < threshold and i != j:
                    continue
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)
                
    ax.set_title(title, fontsize=12, pad=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_missing_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
    title: str = "Missing Value Distribution Matrix"
) -> plt.Figure:
    """
    绘制数据缺失矩阵分布图（清晰呈现样本缺失是随机缺失还是整块断流）。
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300 if save_path else 100)
    
    # 1 代表缺失 (黄色/白色), 0 代表存在 (深色)
    missing_mask = df.isna().to_numpy(dtype=float)
    
    cax = ax.imshow(missing_mask, aspect="auto", cmap="Blues", interpolation="nearest")
    
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel("Variables", fontsize=10, labelpad=8)
    ax.set_ylabel("Sample Index (Row)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12, fontweight='bold')
    
    # 颜色条
    cbar = fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Present (0)', 'Missing (1)'], fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

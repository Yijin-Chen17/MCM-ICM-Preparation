"""
MCM/ICM Preprocessing Toolkit Comprehensive Test Suite
测试数据预处理工具箱 6 大模块的核心功能、抗造性（Defensive Checks）与边界容错。
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# 动态将各模块所在路径加入 sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(BASE_DIR, "01_eda_and_audit"),
    os.path.join(BASE_DIR, "02_strings_and_categorical"),
    os.path.join(BASE_DIR, "03_missing_and_outliers"),
    os.path.join(BASE_DIR, "04_transforms_and_scaling"),
    os.path.join(BASE_DIR, "05_timeseries_and_momentum"),
    os.path.join(BASE_DIR, "06_spatial_and_smoothing"),
])

from eda_toolkit import audit_dataset, detect_constant_columns, export_latex_table1
from category_toolkit import clean_numeric_string, encode_binary, encode_ordinal, encode_onehot, encode_target_cv, export_codebook_latex
from cleaning_toolkit import detect_outliers_iqr, winsorize_iqr, interpolate_series, impute_knn
from transform_toolkit import transform_boxcox, transform_yeojohnson, scale_benefit, scale_cost, scale_interval, normalize_topsis_vector, normalize_minmax_epsilon
from timeseries_toolkit import calc_rolling_features, calc_ewma_momentum, calc_streak_length, calc_lag_diff
from spatial_smooth_toolkit import smooth_savitzky_golay, create_spatial_grid_2d, calc_distance_matrix


class TestPreprocessingToolkit(unittest.TestCase):

    def test_01_eda_and_audit(self):
        # 构建混合脏数据测试体检报告
        df = pd.DataFrame({
            "normal_val": [10.0, 12.0, 11.5, 9.8, 10.5, 11.0],
            "skewed_val": [1.0, 1.2, 1.1, 1.3, 100.0, 1.0],
            "const_val":  [5, 5, 5, 5, 5, 5],
            "missing_val": [np.nan, 2.0, np.nan, 4.0, np.nan, 6.0],
            "category":   ["A", "B", "A", "B", "A", "B"]
        })
        
        report = audit_dataset(df)
        self.assertEqual(len(report), 5)
        
        # 验证常数列识别
        const_cols = detect_constant_columns(df)
        self.assertIn("const_val", const_cols)
        
        # 验证 LaTeX Table 1 导出
        latex = export_latex_table1(df)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("\\toprule", latex)
        self.assertIn("normal\\_val", latex)

    def test_02_strings_and_categorical(self):
        # 1. 脏字符串清洗
        s_dirty = pd.Series(["$1,250.50", "(200.00)", "85.4%", "N/A", "15.2km"])
        s_clean = clean_numeric_string(s_dirty)
        self.assertAlmostEqual(s_clean.iloc[0], 1250.50)
        self.assertAlmostEqual(s_clean.iloc[1], -200.00)
        self.assertAlmostEqual(s_clean.iloc[2], 85.4)
        self.assertTrue(pd.isna(s_clean.iloc[3]))
        self.assertAlmostEqual(s_clean.iloc[4], 15.2)
        
        # 2. 二元映射
        s_bin = pd.Series(["Win", "Loss", "Win", "Win"])
        s_bin_enc = encode_binary(s_bin)
        self.assertEqual(s_bin_enc.tolist(), [1.0, 0.0, 1.0, 1.0])
        
        # 3. 有序编码
        s_ord = pd.Series(["Low", "High", "Medium", "Low"])
        s_ord_enc = encode_ordinal(s_ord, ["Low", "Medium", "High"])
        self.assertEqual(s_ord_enc.tolist(), [0.0, 2.0, 1.0, 0.0])
        
        # 4. 罕见长尾合并独热编码
        df_cat = pd.DataFrame({"Team": ["A", "A", "A", "A", "Rare1", "Rare2"]})
        df_onehot = encode_onehot(df_cat, column="Team", min_freq=0.3, drop_first=False)
        self.assertIn("Team_A", df_onehot.columns)
        self.assertIn("Team_Other", df_onehot.columns)
        self.assertNotIn("Team_Rare1", df_onehot.columns)
        
        # 5. 目标编码 5 折交叉验证
        df_target = pd.DataFrame({
            "Category": ["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"],
            "Target": [1.0, 5.0, 1.2, 5.1, 0.9, 4.8, 1.1, 5.2]
        })
        s_targ = encode_target_cv(df_target, cat_col="Category", target_col="Target", n_splits=2)
        self.assertEqual(len(s_targ), 8)
        self.assertTrue(s_targ.iloc[0] < s_targ.iloc[1]) # C1 均值应显著小于 C2

    def test_03_missing_and_outliers(self):
        s_out = pd.Series([10.0, 10.2, 9.8, 10.1, 10.0, 100.0, -50.0])
        mask = detect_outliers_iqr(s_out, k=1.5)
        self.assertTrue(mask.iloc[5]) # 100.0 应当被标记为异常
        self.assertTrue(mask.iloc[6]) # -50.0 应当被标记为异常
        
        # 缩尾处理 (Winsorization)
        s_wins = winsorize_iqr(s_out, k=1.5)
        self.assertTrue(s_wins.max() < 50.0)
        self.assertTrue(s_wins.min() > -20.0)
        self.assertEqual(len(s_wins), len(s_out)) # 样本量绝不减少
        
        # PCHIP 单调保形插值
        s_gap = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        s_interp = interpolate_series(s_gap, method='pchip')
        self.assertAlmostEqual(s_interp.iloc[2], 3.0, places=1)
        self.assertFalse(s_interp.isna().any())

    def test_04_transforms_and_scaling(self):
        # 1. 效益型正向化
        s_ben = pd.Series([10, 20, 30])
        self.assertEqual(scale_benefit(s_ben).tolist(), [0.0, 0.5, 1.0])
        
        # 2. 成本型正向化
        s_cost = pd.Series([10, 20, 30])
        self.assertEqual(scale_cost(s_cost, method='difference').tolist(), [1.0, 0.5, 0.0])
        
        # 3. 区间型最优映射
        s_int = pd.Series([10, 22, 24, 40]) # 最佳 20~25
        res_int = scale_interval(s_int, optimal_range=(20, 25), tolerance_range=(0, 50))
        self.assertAlmostEqual(res_int.iloc[1], 1.0)
        self.assertAlmostEqual(res_int.iloc[2], 1.0)
        self.assertTrue(res_int.iloc[0] < 1.0)
        self.assertTrue(res_int.iloc[3] < 1.0)
        
        # 4. Box-Cox 自动恢复正态
        s_skew = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 50.0, 200.0])
        s_trans, lmbda, c = transform_boxcox(s_skew)
        self.assertFalse(s_trans.isna().any())

    def test_05_timeseries_and_momentum(self):
        s_pts = pd.Series([1, 1, 1, 0, 1, 1])
        
        # 连击 Streak 统计
        streaks = calc_streak_length(s_pts, target_value=1)
        self.assertEqual(streaks.tolist(), [1, 2, 3, 0, 1, 2])
        
        # EWMA 势头衰减
        momentum = calc_ewma_momentum(s_pts, alpha=0.3)
        self.assertEqual(len(momentum), 6)
        self.assertTrue(momentum.iloc[2] > momentum.iloc[3]) # 连得3分后的势头应高于失分后的势头
        
        # 多尺度滑动窗口
        roll_df = calc_rolling_features(s_pts, windows=[2, 3], metrics=['mean', 'std'])
        self.assertIn("val_roll_2_mean", roll_df.columns)
        self.assertIn("val_roll_3_std", roll_df.columns)

    def test_06_spatial_and_smoothing(self):
        # 1. Savitzky-Golay 保形平滑与解析求导
        xs = np.linspace(0, 10, 50)
        noisy = np.sin(xs) + np.random.normal(0, 0.1, 50)
        # 偶数窗口自适应自愈与求导
        v = smooth_savitzky_golay(pd.Series(noisy), window_length=10, polyorder=2, deriv=1, delta=0.2)
        self.assertEqual(len(v), 50)
        self.assertFalse(v.isna().any())
        
        # 2. 2D 网格剖分
        grid_df, neighbors = create_spatial_grid_2d(x_bounds=(0, 2), y_bounds=(0, 2), step=1.0)
        self.assertEqual(len(grid_df), 4) # 2x2 = 4个格子
        self.assertEqual(len(neighbors[0]), 2) # 0号网格在角落有2个相邻格子
        
        # 3. 空间距离矩阵 (Haversine 经纬度距离)
        coords = np.array([[30.0, 120.0], [30.0, 121.0]])
        dist_m = calc_distance_matrix(coords, metric='haversine')
        self.assertEqual(dist_m.shape, (2, 2))
        self.assertAlmostEqual(dist_m[0, 0], 0.0)
        self.assertTrue(dist_m[0, 1] > 90.0 and dist_m[0, 1] < 105.0) # 纬度30度处经度相差1度约为96公里


if __name__ == "__main__":
    unittest.main()

"""
简化时间预测模块
基于Y染色体浓度与BMI的简化线性模型，预测达标时间
按照新思路：Y浓度 = f(BMI, 孕周)的简化关系
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize_scalar
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TimePredictionModel:
    """基于简化模型的达标时间预测器"""
    
    def __init__(self, data_path: str = None):
        # 达标阈值
        self.threshold = 0.04
        
        # 模型参数（将通过数据拟合获得）
        self.model_params = {}
        self.bmi_correlation = None
        self.fitted_model = None
        
        # 如果提供了数据路径，自动进行拟合
        if data_path:
            self.fit_model_from_data(data_path)
    
    def fit_model_from_data(self, data_path: str):
        """
        从第一问的处理数据中拟合简化模型
        实现Y浓度与BMI相关性分析和线性拟合
        """
        # 加载第一问的处理数据
        data = pd.read_csv(data_path)
        
        print("=== Y染色体浓度与BMI相关性分析 ===")
        
        # 1. 相关性分析
        correlation = data['Y染色体浓度'].corr(data['孕妇BMI'])
        self.bmi_correlation = correlation
        
        # 计算相关性的显著性
        n = len(data.dropna(subset=['Y染色体浓度', '孕妇BMI']))
        t_stat = correlation * np.sqrt((n-2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
        
        print(f"Y浓度与BMI相关系数: {correlation:.4f}")
        print(f"相关性显著性检验: t={t_stat:.4f}, p={p_value:.6f}")
        print(f"样本数量: {n}")
        
        # 2. 拟合简化随机森林模型：Y浓度 = f(孕周, BMI) [无交互项]
        # 保留主要特征但去掉交互项，使用保守参数避免过拟合
        X = data[['gestational_days', '孕妇BMI']].copy().dropna()
        y = data.loc[X.index, 'Y染色体浓度']
        
        # 数据划分（用于更准确的性能评估）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 使用保守的随机森林参数避免过拟合
        model = RandomForestRegressor(
            n_estimators=30,       # 进一步减少树的数量
            max_depth=4,           # 限制更小的树深度
            min_samples_split=20,  # 增加分割所需的最小样本数
            min_samples_leaf=10,   # 增加叶子节点最小样本数
            max_features='sqrt',   # 使用sqrt特征数
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.fitted_model = model
        self.feature_names = X.columns.tolist()
        
        # 保存模型参数（随机森林的特征重要性）
        self.model_params = {
            'feature_importance': dict(zip(self.feature_names, model.feature_importances_)),
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf
        }
        
        # 模型性能评估
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"\n=== 随机森林模型拟合结果 ===")
        print(f"模型类型: RandomForest (n_estimators={model.n_estimators}, max_depth={model.max_depth})")
        print("特征重要性:")
        for feature, importance in self.model_params['feature_importance'].items():
            print(f"  - {feature}: {importance:.4f}")
        print(f"训练集 R² = {r2_train:.4f}")
        print(f"测试集 R² = {r2_test:.4f}")
        print(f"训练集 RMSE = {rmse_train:.6f}")
        print(f"测试集 RMSE = {rmse_test:.6f}")
        
        return self.model_params
    def predict_concentration(self, gestational_days: float, bmi: float, 
                            **kwargs) -> float:
        """
        使用简化随机森林模型预测Y染色体浓度
        简化后的模型使用孕周和BMI两个主要特征（无交互项）
        
        Parameters:
        - gestational_days: 孕周天数
        - bmi: BMI值
        """
        if self.fitted_model is None:
            raise ValueError("模型未拟合！请先调用fit_model_from_data()或提供数据路径")
        
        # 使用孕周和BMI两个特征（无交互项）
        features = pd.DataFrame({
            'gestational_days': [gestational_days],
            '孕妇BMI': [bmi]
        })
        
        # 使用随机森林模型预测浓度
        concentration = self.fitted_model.predict(features)[0]
        
        return max(0, concentration)  # 浓度不能为负
    
    def solve_达标时间(self, bmi: float, **kwargs) -> Optional[float]:
        """
        求解Y染色体浓度达到4%的最早时间
        简化版本只需要BMI参数
        """
        if self.fitted_model is None:
            raise ValueError("模型未拟合！请先调用fit_model_from_data()或提供数据路径")
        
        def concentration_equation(t):
            # Y浓度方程：target_concentration - predicted_concentration = 0
            return self.threshold - self.predict_concentration(t, bmi)
        
        try:
            # 使用数值求解找到达标时间
            # 搜索范围：50-300天（约7-43周）
            from scipy.optimize import brentq
            
            # 检查边界条件
            if concentration_equation(50) <= 0:
                return 50.0  # 已经达标
            
            if concentration_equation(300) > 0:
                return None  # 在合理时间内无法达标
                
            # 使用Brent方法求根
            solution = brentq(concentration_equation, 50, 300, xtol=0.1)
            return solution
            
        except Exception as e:
            print(f"求解达标时间失败: {e}")
            return None
    
    def predict_达标概率(self, gestational_days: float, bmi: float,
                       measurement_error: float = 0.005,
                       individual_effect_std: float = 0.01) -> float:
        """
        预测在给定时间点达标的概率（考虑测量误差和个体差异）
        简化版本只需要gestational_days和bmi参数
        
        Parameters:
        - gestational_days: 孕周天数
        - bmi: BMI值
        - measurement_error: 测量标准误差
        - individual_effect_std: 个体差异标准差
        """
        predicted_conc = self.predict_concentration(gestational_days, bmi)
        
        # 总方差 = 测量误差方差 + 个体差异方差
        total_std = np.sqrt(measurement_error**2 + individual_effect_std**2)
        
        # 计算Y浓度 ≥ 4%的概率
        from scipy.stats import norm
        prob = 1 - norm.cdf(self.threshold, predicted_conc, total_std)
        
        return prob
    
    def find_time_for_success_probability(self, bmi: float, target_prob: float = 0.95,
                                        measurement_error: float = 0.005) -> Optional[float]:
        """
        找到达到目标成功概率的最早时间
        简化版本只需要BMI参数
        
        Parameters:
        - bmi: BMI值
        - target_prob: 目标成功概率（默认95%）
        - measurement_error: 测量误差
        
        Returns:
        - gestational_days: 达到目标概率的最早孕周天数
        """
        def prob_equation(t):
            """概率方程，求解 P(Y浓度 ≥ 4%) - target_prob = 0"""
            prob = self.predict_达标概率(t, bmi, measurement_error)
            return prob - target_prob
        
        # 在合理的孕周范围内搜索解
        min_gd, max_gd = 70, 220  # 10-31周
        
        try:
            # 检查边界条件
            prob_min = prob_equation(min_gd)
            prob_max = prob_equation(max_gd)
            
            if prob_min >= 0:
                # 最早就能满足概率要求
                return min_gd
            elif prob_max < 0:
                # 在合理时间内无法满足概率要求
                return None
            else:
                # 在范围内求解
                from scipy.optimize import brentq
                solution = brentq(prob_equation, min_gd, max_gd, xtol=0.1)
                return solution
                
        except Exception as e:
            print(f"求解95%概率约束时间失败: {e}")
            return None
    
    def batch_predict_达标时间(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        批量预测达标时间
        简化版本只需要BMI信息
        """
        results = []
        
        for _, row in data.iterrows():
            # 使用简化模型预测达标时间
            达标时间 = self.solve_达标时间(bmi=row['孕妇BMI'])
            
            # 计算95%概率约束时间
            约束时间 = self.find_time_for_success_probability(bmi=row['孕妇BMI'])
            
            results.append({
                '孕妇代码': row['孕妇代码'],
                '孕妇BMI': row['孕妇BMI'],
                '预测达标时间': 达标时间,
                '预测达标周数': 达标时间/7 if 达标时间 else None,
                '95%约束时间': 约束时间,
                '95%约束周数': 约束时间/7 if 约束时间 else None
            })
            
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试模块 - 需要先拟合模型
    data_path = "../problem1_analysis/results/data/processed_data.csv"
    predictor = TimePredictionModel(data_path)
    
    # 测试单个预测
    test_达标时间 = predictor.solve_达标时间(bmi=30)
    print(f"测试达标时间: {test_达标时间}天 ({test_达标时间/7:.1f}周)")
    
    # 测试概率预测
    prob = predictor.predict_达标概率(120, bmi=30)
    print(f"120天时达标概率: {prob:.3f}")
    
    # 测试95%约束时间
    constraint_time = predictor.find_time_for_success_probability(bmi=30)
    print(f"95%约束时间: {constraint_time}天 ({constraint_time/7:.1f}周)")
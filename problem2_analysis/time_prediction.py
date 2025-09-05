"""
时间预测模块
基于第一问的混合效应模型，预测Y染色体浓度达标时间
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize_scalar
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TimePredictionModel:
    """基于第一问模型的达标时间预测器"""
    
    def __init__(self):
        # 第一问得到的最优模型参数
        self.model_params = {
            'intercept': -0.248662,
            'gd_linear': -0.000797,
            'gd_quadratic': 0.000005,
            'bmi_centered': 0.011867,
            'age_centered': -0.000570,
            'height': 0.004788,
            'weight': -0.005092
        }
        
        # 中心化参数（从第一问数据计算得出）
        self.center_params = {
            'bmi_mean': 32.32,
            'age_mean': 30.8
        }
        
        # 达标阈值
        self.threshold = 0.04
        
    def predict_concentration(self, gestational_days: float, bmi: float, 
                            age: float, height: float, weight: float,
                            individual_effect: float = 0.0) -> float:
        """
        预测Y染色体浓度
        
        Parameters:
        - gestational_days: 孕周天数
        - bmi: BMI值
        - age: 年龄
        - height: 身高
        - weight: 体重
        - individual_effect: 个体随机效应
        """
        # 中心化处理
        bmi_c = bmi - self.center_params['bmi_mean']
        age_c = age - self.center_params['age_mean']
        
        # 计算浓度
        concentration = (
            self.model_params['intercept'] +
            self.model_params['gd_linear'] * gestational_days +
            self.model_params['gd_quadratic'] * (gestational_days ** 2) +
            self.model_params['bmi_centered'] * bmi_c +
            self.model_params['age_centered'] * age_c +
            self.model_params['height'] * height +
            self.model_params['weight'] * weight +
            individual_effect
        )
        
        return max(0, concentration)  # 浓度不能为负
    
    def solve_达标时间(self, bmi: float, age: float, height: float, 
                     weight: float, individual_effect: float = 0.0) -> Optional[float]:
        """
        求解Y染色体浓度达到4%的最早时间
        
        Returns:
        - gestational_days: 达标的孕周天数，如果无法达标返回None
        """
        
        def concentration_equation(gd):
            """浓度方程，求解 concentration - threshold = 0"""
            pred_conc = self.predict_concentration(gd, bmi, age, height, weight, individual_effect)
            return pred_conc - self.threshold
        
        # 在合理的孕周范围内搜索解
        min_gd, max_gd = 70, 200  # 10-28周
        
        try:
            # 检查边界条件
            conc_min = concentration_equation(min_gd)
            conc_max = concentration_equation(max_gd)
            
            if conc_min >= 0:
                # 最早就能达标
                return min_gd
            elif conc_max < 0:
                # 在合理时间内无法达标
                return None
            else:
                # 在范围内求解
                solution = fsolve(concentration_equation, (min_gd + max_gd) / 2)[0]
                return max(min_gd, min(max_gd, solution))
                
        except:
            return None
    
    def predict_达标概率(self, gestational_days: float, bmi: float, 
                       age: float, height: float, weight: float,
                       measurement_error: float = 0.005,
                       individual_effect_std: float = 0.01) -> float:
        """
        预测在给定时间点达标的概率（考虑测量误差和个体差异）
        
        Parameters:
        - measurement_error: 测量标准误差
        - individual_effect_std: 个体差异标准差
        """
        predicted_conc = self.predict_concentration(gestational_days, bmi, age, height, weight)
        
        # 总方差 = 测量误差方差 + 个体差异方差
        total_std = np.sqrt(measurement_error**2 + individual_effect_std**2)
        
        # 计算Y浓度 ≥ 4%的概率
        from scipy.stats import norm
        prob = 1 - norm.cdf(self.threshold, predicted_conc, total_std)
        
        return prob
    
    def find_time_for_success_probability(self, bmi: float, age: float, height: float, 
                                        weight: float, target_prob: float = 0.95,
                                        measurement_error: float = 0.005) -> Optional[float]:
        """
        找到达到目标成功概率的最早时间
        
        Parameters:
        - target_prob: 目标成功概率（默认95%）
        
        Returns:
        - gestational_days: 达到目标概率的最早孕周天数
        """
        def prob_equation(t):
            """概率方程，求解 P(Y浓度 ≥ 4%) - target_prob = 0"""
            prob = self.predict_达标概率(t, bmi, age, height, weight, measurement_error)
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
                from scipy.optimize import fsolve
                solution = fsolve(prob_equation, (min_gd + max_gd) / 2)[0]
                return max(min_gd, min(max_gd, solution))
                
        except:
            return None
    
    def batch_predict_达标时间(self, data: pd.DataFrame) -> pd.DataFrame:
        """批量预测达标时间"""
        results = []
        
        for _, row in data.iterrows():
            达标时间 = self.solve_达标时间(
                bmi=row['孕妇BMI'],
                age=row['年龄'],  
                height=row['身高'],
                weight=row['体重']
            )
            
            results.append({
                '孕妇代码': row['孕妇代码'],
                '孕妇BMI': row['孕妇BMI'],
                '年龄': row['年龄'],
                '身高': row['身高'],
                '体重': row['体重'],
                '预测达标时间': 达标时间,
                '预测达标周数': 达标时间/7 if 达标时间 else None
            })
            
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试模块
    predictor = TimePredictionModel()
    
    # 测试单个预测
    test_达标时间 = predictor.solve_达标时间(bmi=30, age=28, height=165, weight=70)
    print(f"测试达标时间: {test_达标时间}天 ({test_达标时间/7:.1f}周)")
    
    # 测试概率预测
    prob = predictor.predict_达标概率(120, bmi=30, age=28, height=165, weight=70)
    print(f"120天时达标概率: {prob:.3f}")
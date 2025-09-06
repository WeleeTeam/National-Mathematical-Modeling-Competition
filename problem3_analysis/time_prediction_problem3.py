"""
问题3时间预测模型
基于第一问的最新混合效应模型结果，考虑多因素影响
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class TimePredictionModelProblem3:
    """问题3时间预测模型 - 基于第一问增强混合效应模型"""
    
    def __init__(self):
        # 基于第一问增强混合效应模型的最优参数
        self.model_params = {
            'intercept': 0.044276,
            'gestational_days': -0.000568,
            'bmi_centered': 0.000459,
            'gd_bmi_interaction': 0.000016,
            'age_centered': -0.000537,
            'gd_age_interaction': -0.000002,
            'height': 0.001113,
            'weight': -0.001463,
            'x_concentration': -0.495231,
            'y_z_score': -0.013853,
            'x_z_score': -0.000160,
            'chr18_z_score': -0.000002,
            'blood_draw_num': 0.010511,
            'gd_y_z_interaction': 0.000129,
            'gd_x_conc_interaction': 0.006594
        }
        
        # 中心化参数（基于第一问数据）
        self.center_params = {
            'bmi_mean': 32.32,
            'age_mean': 29.5,
            'height_mean': 160.0,
            'weight_mean': 80.0
        }
        
        # 阈值参数
        self.threshold = 0.04  # Y染色体浓度达标阈值4%
        
        # 测量误差参数
        self.measurement_error_std = 0.005
        self.individual_effect_std = 0.01
        
        print("问题3时间预测模型初始化完成")
        print(f"使用第一问增强混合效应模型参数")
        print(f"模型包含{len(self.model_params)}个参数")
    
    def predict_concentration(self, gestational_days: float, bmi: float, 
                            age: float, height: float, weight: float,
                            x_concentration: float = 0.5,  # 默认X染色体浓度
                            y_z_score: float = 0.0,       # 默认Y染色体Z值
                            x_z_score: float = 0.0,       # 默认X染色体Z值
                            chr18_z_score: float = 0.0,   # 默认18号染色体Z值
                            blood_draw_num: int = 1,      # 默认抽血次数
                            individual_effect: float = 0.0) -> float:
        """
        预测Y染色体浓度（基于第一问增强混合效应模型）
        
        Parameters:
        - gestational_days: 孕周天数
        - bmi: 孕妇BMI
        - age: 孕妇年龄
        - height: 孕妇身高
        - weight: 孕妇体重
        - x_concentration: X染色体浓度
        - y_z_score: Y染色体Z值
        - x_z_score: X染色体Z值
        - chr18_z_score: 18号染色体Z值
        - blood_draw_num: 抽血次数
        - individual_effect: 个体随机效应
        
        Returns:
        - predicted_concentration: 预测的Y染色体浓度
        """
        # 中心化变量
        bmi_centered = bmi - self.center_params['bmi_mean']
        age_centered = age - self.center_params['age_mean']
        
        # 计算预测值
        concentration = (
            self.model_params['intercept'] +
            self.model_params['gestational_days'] * gestational_days +
            self.model_params['bmi_centered'] * bmi_centered +
            self.model_params['gd_bmi_interaction'] * gestational_days * bmi_centered +
            self.model_params['age_centered'] * age_centered +
            self.model_params['gd_age_interaction'] * gestational_days * age_centered +
            self.model_params['height'] * height +
            self.model_params['weight'] * weight +
            self.model_params['x_concentration'] * x_concentration +
            self.model_params['y_z_score'] * y_z_score +
            self.model_params['x_z_score'] * x_z_score +
            self.model_params['chr18_z_score'] * chr18_z_score +
            self.model_params['blood_draw_num'] * blood_draw_num +
            self.model_params['gd_y_z_interaction'] * gestational_days * y_z_score +
            self.model_params['gd_x_conc_interaction'] * gestational_days * x_concentration +
            individual_effect
        )
        
        return max(0, concentration)  # 确保非负
    
    def solve_达标时间(self, bmi: float, age: float, height: float, 
                      weight: float, x_concentration: float = 0.5,
                      y_z_score: float = 0.0, x_z_score: float = 0.0,
                      chr18_z_score: float = 0.0, blood_draw_num: int = 1,
                      individual_effect: float = 0.0) -> Optional[float]:
        """
        求解Y染色体浓度达到4%的最早时间
        
        Parameters:
        - bmi: 孕妇BMI
        - age: 孕妇年龄
        - height: 孕妇身高
        - weight: 孕妇体重
        - x_concentration: X染色体浓度
        - y_z_score: Y染色体Z值
        - x_z_score: X染色体Z值
        - chr18_z_score: 18号染色体Z值
        - blood_draw_num: 抽血次数
        - individual_effect: 个体随机效应
        
        Returns:
        - optimal_time: 达标时间（天数），如果无法达标则返回None
        """
        from scipy.optimize import fsolve
        
        def concentration_equation(gd):
            return self.predict_concentration(
                gd, bmi, age, height, weight, x_concentration,
                y_z_score, x_z_score, chr18_z_score, blood_draw_num, individual_effect
            ) - self.threshold
        
        try:
            # 在合理范围内搜索
            result = fsolve(concentration_equation, x0=70, full_output=True)
            if result[2] == 1:  # 求解成功
                optimal_time = result[0][0]
                if 50 <= optimal_time <= 200:  # 合理范围检查
                    return optimal_time
            return None
        except:
            return None
    
    def predict_达标概率(self, gestational_days: float, bmi: float, 
                        age: float, height: float, weight: float,
                        x_concentration: float = 0.5,
                        y_z_score: float = 0.0, x_z_score: float = 0.0,
                        chr18_z_score: float = 0.0, blood_draw_num: int = 1,
                        measurement_error: float = 0.005,
                        individual_effect_std: float = 0.01) -> float:
        """
        预测在指定时间点Y染色体浓度达标的概率
        
        Parameters:
        - gestational_days: 孕周天数
        - bmi: 孕妇BMI
        - age: 孕妇年龄
        - height: 孕妇身高
        - weight: 孕妇体重
        - x_concentration: X染色体浓度
        - y_z_score: Y染色体Z值
        - x_z_score: X染色体Z值
        - chr18_z_score: 18号染色体Z值
        - blood_draw_num: 抽血次数
        - measurement_error: 测量误差标准差
        - individual_effect_std: 个体效应标准差
        
        Returns:
        - success_probability: 达标概率
        """
        from scipy.stats import norm
        
        # 预测浓度
        predicted_conc = self.predict_concentration(
            gestational_days, bmi, age, height, weight, x_concentration,
            y_z_score, x_z_score, chr18_z_score, blood_draw_num
        )
        
        # 计算总误差方差
        total_variance = measurement_error**2 + individual_effect_std**2
        
        # 计算达标概率
        z_score = (self.threshold - predicted_conc) / np.sqrt(total_variance)
        success_probability = 1 - norm.cdf(z_score)
        
        return success_probability
    
    def find_time_for_success_probability(self, bmi: float, age: float, height: float, 
                                        weight: float, target_prob: float = 0.95,
                                        x_concentration: float = 0.5,
                                        y_z_score: float = 0.0, x_z_score: float = 0.0,
                                        chr18_z_score: float = 0.0, blood_draw_num: int = 1,
                                        measurement_error: float = 0.005) -> Optional[float]:
        """
        找到满足指定成功概率的最早时间
        
        Parameters:
        - bmi: 孕妇BMI
        - age: 孕妇年龄
        - height: 孕妇身高
        - weight: 孕妇体重
        - target_prob: 目标成功概率
        - x_concentration: X染色体浓度
        - y_z_score: Y染色体Z值
        - x_z_score: X染色体Z值
        - chr18_z_score: 18号染色体Z值
        - blood_draw_num: 抽血次数
        - measurement_error: 测量误差标准差
        
        Returns:
        - optimal_time: 满足概率要求的最早时间，如果无法满足则返回None
        """
        from scipy.optimize import fsolve
        
        def prob_equation(t):
            return self.predict_达标概率(
                t, bmi, age, height, weight, x_concentration,
                y_z_score, x_z_score, chr18_z_score, blood_draw_num, measurement_error
            ) - target_prob
        
        try:
            # 在合理范围内搜索
            result = fsolve(prob_equation, x0=70, full_output=True)
            if result[2] == 1:  # 求解成功
                optimal_time = result[0][0]
                if 50 <= optimal_time <= 200:  # 合理范围检查
                    return optimal_time
            return None
        except:
            return None
    
    def batch_predict_达标时间(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        批量预测达标时间
        
        Parameters:
        - data: 包含孕妇信息的DataFrame
        
        Returns:
        - results: 包含预测结果的DataFrame
        """
        results = []
        
        for _, patient in data.iterrows():
            # 提取基本信息
            bmi = patient['孕妇BMI']
            age = patient['年龄']
            height = patient['身高']
            weight = patient['体重']
            
            # 提取额外信息（如果存在）
            x_concentration = patient.get('X染色体浓度', 0.5)
            y_z_score = patient.get('Y染色体Z值', 0.0)
            x_z_score = patient.get('X染色体Z值', 0.0)
            chr18_z_score = patient.get('18号染色体Z值', 0.0)
            blood_draw_num = patient.get('检测抽血次数', 1)
            
            # 预测达标时间
            optimal_time = self.solve_达标时间(
                bmi, age, height, weight, x_concentration,
                y_z_score, x_z_score, chr18_z_score, blood_draw_num
            )
            
            # 预测95%概率约束时间
            constraint_time = self.find_time_for_success_probability(
                bmi, age, height, weight, 0.95, x_concentration,
                y_z_score, x_z_score, chr18_z_score, blood_draw_num
            )
            
            results.append({
                '孕妇代码': patient['孕妇代码'],
                '预测达标时间': optimal_time,
                '预测达标周数': optimal_time / 7 if optimal_time else None,
                '满足95%约束时间': constraint_time,
                '满足95%约束周数': constraint_time / 7 if constraint_time else None,
                'BMI': bmi,
                '年龄': age,
                '身高': height,
                '体重': weight,
                'X染色体浓度': x_concentration,
                'Y染色体Z值': y_z_score,
                'X染色体Z值': x_z_score,
                '18号染色体Z值': chr18_z_score,
                '检测抽血次数': blood_draw_num
            })
        
        return pd.DataFrame(results)
    
    def calculate_multifactor_score(self, bmi: float, age: float, height: float, 
                                  weight: float, x_concentration: float = 0.5,
                                  y_z_score: float = 0.0, x_z_score: float = 0.0,
                                  chr18_z_score: float = 0.0, blood_draw_num: int = 1) -> Dict:
        """
        计算多因素综合评分（用于分组）
        
        Parameters:
        - bmi: 孕妇BMI
        - age: 孕妇年龄
        - height: 孕妇身高
        - weight: 孕妇体重
        - x_concentration: X染色体浓度
        - y_z_score: Y染色体Z值
        - x_z_score: X染色体Z值
        - chr18_z_score: 18号染色体Z值
        - blood_draw_num: 抽血次数
        
        Returns:
        - score_dict: 包含各项评分的字典
        """
        # 基础因素评分
        bmi_score = bmi / 50.0  # 标准化BMI评分
        age_score = age / 50.0  # 标准化年龄评分
        height_score = height / 200.0  # 标准化身高评分
        weight_score = weight / 150.0  # 标准化体重评分
        
        # 检测质量评分
        quality_score = blood_draw_num / 5.0  # 抽血次数评分
        
        # 染色体指标评分
        x_conc_score = x_concentration / 1.0  # X染色体浓度评分
        y_z_score_norm = abs(y_z_score) / 3.0  # Y染色体Z值评分
        x_z_score_norm = abs(x_z_score) / 3.0  # X染色体Z值评分
        chr18_z_score_norm = abs(chr18_z_score) / 3.0  # 18号染色体Z值评分
        
        # 综合评分（加权平均）
        weights = {
            'bmi': 0.25,
            'age': 0.15,
            'height': 0.10,
            'weight': 0.15,
            'quality': 0.15,
            'x_conc': 0.10,
            'y_z': 0.05,
            'x_z': 0.02,
            'chr18_z': 0.03
        }
        
        total_score = (
            weights['bmi'] * bmi_score +
            weights['age'] * age_score +
            weights['height'] * height_score +
            weights['weight'] * weight_score +
            weights['quality'] * quality_score +
            weights['x_conc'] * x_conc_score +
            weights['y_z'] * y_z_score_norm +
            weights['x_z'] * x_z_score_norm +
            weights['chr18_z'] * chr18_z_score_norm
        )
        
        return {
            'total_score': total_score,
            'bmi_score': bmi_score,
            'age_score': age_score,
            'height_score': height_score,
            'weight_score': weight_score,
            'quality_score': quality_score,
            'x_conc_score': x_conc_score,
            'y_z_score_norm': y_z_score_norm,
            'x_z_score_norm': x_z_score_norm,
            'chr18_z_score_norm': chr18_z_score_norm,
            'weights': weights
        }


if __name__ == "__main__":
    # 测试问题3时间预测模型
    print("问题3时间预测模型测试")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        '孕妇代码': ['T001', 'T002', 'T003'],
        '孕妇BMI': [25.0, 32.0, 40.0],
        '年龄': [28, 30, 35],
        '身高': [160, 165, 170],
        '体重': [60, 80, 100],
        'X染色体浓度': [0.5, 0.5, 0.5],
        'Y染色体Z值': [0.0, 0.0, 0.0],
        'X染色体Z值': [0.0, 0.0, 0.0],
        '18号染色体Z值': [0.0, 0.0, 0.0],
        '检测抽血次数': [1, 1, 1]
    })
    
    model = TimePredictionModelProblem3()
    
    # 测试批量预测
    results = model.batch_predict_达标时间(test_data)
    print("\n批量预测结果:")
    print(results[['孕妇代码', '预测达标周数', '满足95%约束周数']])
    
    # 测试多因素评分
    print("\n多因素评分测试:")
    for _, patient in test_data.iterrows():
        score = model.calculate_multifactor_score(
            patient['孕妇BMI'], patient['年龄'], patient['身高'], patient['体重']
        )
        print(f"患者 {patient['孕妇代码']}: 综合评分 = {score['total_score']:.3f}")
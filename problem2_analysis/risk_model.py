"""
风险模型模块
构建孕妇潜在风险函数和检测误差概率模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RiskModel:
    """孕妇潜在风险评估模型 - 按照概率约束和连续风险函数设计"""
    
    def __init__(self):
        # 概率约束参数
        self.target_success_probability = 0.95  # 95%成功概率要求 (p₀)
        self.measurement_error_std = 0.005       # Y染色体浓度测量标准误差
        self.individual_effect_std = 0.01        # 个体差异标准差
        
        # 约束条件参数
        self.bmi_min = 20.0      # BMI最小值 (b_min)
        self.bmi_max = 50.0      # BMI最大值 (b_max)
        self.detection_min_week = 10  # 检测时窗最小值 (10周)
        self.detection_max_week = 25  # 检测时窗最大值 (25周)
        self.detection_min_days = self.detection_min_week * 7
        self.detection_max_days = self.detection_max_week * 7
        
        # 延误诊断风险函数参数
        self.delay_risk_params = {
            'early_base': 0.1,      # 早期基础风险
            'early_slope': 0.02,    # 早期风险斜率  
            'medium_base': 0.5,     # 中期基础风险
            'medium_slope': 0.08,   # 中期风险斜率
            'late_base': 2.0,       # 晚期基础风险
            'late_slope': 0.2       # 晚期风险斜率
        }
        
        # 检测失败惩罚系数
        self.detection_failure_penalty = 10.0
    
    def calculate_delay_risk(self, gestational_days: float) -> float:
        """
        计算延误诊断风险（连续函数）
        
        Parameters:
        - gestational_days: 孕周天数
        
        Returns:
        - delay_risk: 延误风险分数
        """
        params = self.delay_risk_params
        
        if gestational_days <= 84:  # ≤12周，早期
            # 线性增长，但基础风险较低
            normalized_time = (gestational_days - 70) / 14  # 标准化到[0,1]
            delay_risk = params['early_base'] + params['early_slope'] * normalized_time
            
        elif gestational_days <= 189:  # 13-27周，中期
            # 加速增长
            normalized_time = (gestational_days - 84) / 105  # 标准化到[0,1]
            delay_risk = params['medium_base'] + params['medium_slope'] * normalized_time
            
        else:  # ≥28周，晚期
            # 快速增长，高风险
            normalized_time = min((gestational_days - 189) / 21, 1)  # 标准化，但有上限
            delay_risk = params['late_base'] + params['late_slope'] * normalized_time
        
        return max(0, delay_risk)  # 确保风险非负

    def calculate_detection_failure_risk(self, success_probability: float) -> float:
        """
        计算检测失败风险
        
        Parameters:
        - success_probability: 检测成功概率 P(Y浓度 ≥ 4%)
        
        Returns:
        - failure_risk: 检测失败风险
        """
        if success_probability >= self.target_success_probability:
            # 满足95%概率要求，无检测失败风险
            return 0.0
        else:
            # 未满足概率要求，施加惩罚
            probability_deficit = self.target_success_probability - success_probability
            failure_risk = self.detection_failure_penalty * probability_deficit
            return failure_risk
    
    def check_bmi_segmentation_constraints(self, bmi_groups: List[Tuple[float, float]]) -> bool:
        """
        检查BMI分段约束：覆盖、相邻与不交约束
        
        Parameters:
        - bmi_groups: BMI分组列表，每个元素为(下界, 上界)
        
        Returns:
        - is_valid: 是否满足约束条件
        """
        if not bmi_groups:
            return False
        
        # 1. 检查覆盖约束：I₁ ∪ I₂ ∪ ... ∪ I_G = [b_min, b_max]
        sorted_groups = sorted(bmi_groups, key=lambda x: x[0])
        
        # 检查是否覆盖整个BMI范围
        if sorted_groups[0][0] > self.bmi_min or sorted_groups[-1][1] < self.bmi_max:
            return False
        
        # 2. 检查相邻与不交约束：β₀ < β₁ < ... < β_G
        for i in range(len(sorted_groups) - 1):
            current_upper = sorted_groups[i][1]
            next_lower = sorted_groups[i + 1][0]
            
            # 相邻约束：当前组上界 = 下一组下界
            if abs(current_upper - next_lower) > 1e-6:
                return False
        
        # 3. 检查单调性：β₀ < β₁ < ... < β_G
        boundaries = [sorted_groups[0][0]] + [group[1] for group in sorted_groups]
        for i in range(len(boundaries) - 1):
            if boundaries[i] >= boundaries[i + 1]:
                return False
        
        return True
    
    def check_detection_time_constraints(self, detection_times: List[float]) -> bool:
        """
        检查检测时窗约束：10 ≤ τ_g ≤ 25周
        
        Parameters:
        - detection_times: 各组的检测时间（周）
        
        Returns:
        - is_valid: 是否满足约束条件
        """
        for tau_g in detection_times:
            if not (self.detection_min_week <= tau_g <= self.detection_max_week):
                return False
        return True
    
    def check_within_group_robust_constraint(self, bmi_group_data: pd.DataFrame, 
                                           detection_time_weeks: float, 
                                           time_predictor) -> Tuple[bool, float]:
        """
        检查组内稳妥达标约束：inf π(τ_g, b) ≥ p₀
        
        Parameters:
        - bmi_group_data: BMI组数据
        - detection_time_weeks: 该组的检测时间（周）
        - time_predictor: 时间预测模型
        
        Returns:
        - (is_valid, min_probability): 是否满足约束，组内最小概率
        """
        detection_time_days = detection_time_weeks * 7
        success_probabilities = []
        
        for _, patient in bmi_group_data.iterrows():
            prob = time_predictor.predict_达标概率(
                detection_time_days, patient['孕妇BMI'],
                self.measurement_error_std, self.individual_effect_std
            )
            success_probabilities.append(prob)
        
        # 计算组内最小概率（最坏情况）
        min_probability = min(success_probabilities) if success_probabilities else 0
        
        # 检查是否满足约束：inf π(τ_g, b) ≥ p₀
        is_valid = min_probability >= self.target_success_probability
        
        return is_valid, min_probability
    
    def calculate_total_risk(self, gestational_days: float, bmi: float, 
                           time_predictor) -> Dict:
        """
        计算总风险函数 = 检测失败风险 + 延误诊断风险
        简化版本：只需要gestational_days、bmi和time_predictor参数
        
        Parameters:
        - gestational_days: 孕周天数
        - bmi: BMI值
        - time_predictor: 时间预测模型实例
        
        Returns:
        - risk_breakdown: 风险分解结果
        """
        # 1. 计算该时间点的检测成功概率（简化版本）
        success_prob = time_predictor.predict_达标概率(
            gestational_days, bmi,
            self.measurement_error_std, self.individual_effect_std
        )
        
        # 2. 计算检测失败风险
        detection_failure_risk = self.calculate_detection_failure_risk(success_prob)
        
        # 3. 计算延误诊断风险
        delay_risk = self.calculate_delay_risk(gestational_days)
        
        # 4. 总风险
        total_risk = detection_failure_risk + delay_risk
        
        return {
            'total_risk': total_risk,
            'detection_failure_risk': detection_failure_risk,
            'delay_risk': delay_risk,
            'success_probability': success_prob,
            'gestational_days': gestational_days,
            'gestational_weeks': gestational_days / 7,
            'satisfies_constraint': success_prob >= self.target_success_probability
        }
    
    
    def calculate_detection_error_risk(self, predicted_concentration: float, 
                                     actual_test_time: float) -> Dict:
        """
        计算检测误差带来的风险
        
        Parameters:
        - predicted_concentration: 预测的Y染色体浓度
        - actual_test_time: 实际检测时间
        
        Returns:
        - error_risks: 包含各种误差风险的字典
        """
        from scipy.stats import norm
        
        threshold = 0.04
        
        # 计算真实达标但检测为未达标的概率（假阴性）
        if predicted_concentration >= threshold:
            z_score_fn = (threshold - predicted_concentration) / self.measurement_error_std
            false_negative_prob = norm.cdf(z_score_fn)
        else:
            false_negative_prob = 0
        
        # 计算真实未达标但检测为达标的概率（假阳性）  
        if predicted_concentration < threshold:
            z_score_fp = (predicted_concentration - threshold) / self.measurement_error_std
            false_positive_prob = norm.cdf(z_score_fp)
        else:
            false_positive_prob = 0
            
        # 假阴性风险：需要重新检测的延误风险
        fn_risk = false_negative_prob * self.calculate_delay_risk(actual_test_time + 14)  # 假设重检需要2周
        
        # 假阳性风险：漏诊的医疗风险
        fp_risk = false_positive_prob * 5  # 假阳性带来固定风险分数
        
        return {
            'false_negative_prob': false_negative_prob,
            'false_positive_prob': false_positive_prob,
            'false_negative_risk': fn_risk,
            'false_positive_risk': fp_risk,
            'total_error_risk': fn_risk + fp_risk
        }
    
    def calculate_expected_risk(self, test_time: float, bmi_group_data: pd.DataFrame,
                              time_predictor) -> Dict:
        """
        计算给定检测时间下BMI组的期望风险（使用新的总风险函数）
        
        Parameters:
        - test_time: 建议的检测时间（孕周天数）
        - bmi_group_data: 该BMI组的数据
        - time_predictor: 时间预测模型实例
        
        Returns:
        - expected_risks: 期望风险分析结果
        """
        total_risks = []
        detection_failure_risks = []
        delay_risks = []
        success_probabilities = []
        constraint_satisfactions = []
        
        for _, patient in bmi_group_data.iterrows():
            # 使用简化的总风险函数
            risk_breakdown = self.calculate_total_risk(
                test_time, patient['孕妇BMI'], time_predictor
            )
            
            total_risks.append(risk_breakdown['total_risk'])
            detection_failure_risks.append(risk_breakdown['detection_failure_risk'])
            delay_risks.append(risk_breakdown['delay_risk'])
            success_probabilities.append(risk_breakdown['success_probability'])
            constraint_satisfactions.append(risk_breakdown['satisfies_constraint'])
        
        # 计算组统计结果
        expected_risk = np.mean(total_risks)
        success_rate = np.mean(success_probabilities)
        constraint_satisfaction_rate = np.mean(constraint_satisfactions)
        
        return {
            'expected_risk': expected_risk,
            'success_rate': success_rate,
            'constraint_satisfaction_rate': constraint_satisfaction_rate,
            'sample_size': len(bmi_group_data),
            'average_detection_failure_risk': np.mean(detection_failure_risks),
            'average_delay_risk': np.mean(delay_risks),
            'success_rate_min': np.min(success_probabilities),
            'success_rate_max': np.max(success_probabilities),
            'test_time_weeks': test_time / 7,
            'individual_risks': total_risks,
            'individual_success_probs': success_probabilities
        }
    
    def optimize_test_time_for_group(self, bmi_group_data: pd.DataFrame, 
                                   time_predictor, min_week=None, max_week=None) -> Dict:
        """
        为单个BMI组优化检测时间（基于三个约束条件）
        改进版本：优先满足组内稳妥达标约束
        
        Parameters:
        - bmi_group_data: BMI组数据
        - time_predictor: 时间预测模型
        - min_week, max_week: 检测时间范围（周），默认使用约束条件中的范围
        
        Returns:
        - optimization_result: 优化结果
        """
        from scipy.optimize import minimize_scalar
        import numpy as np
        
        # 使用约束条件中的时间范围
        if min_week is None:
            min_week = self.detection_min_week
        if max_week is None:
            max_week = self.detection_max_week
        
        # 首先找到满足组内稳妥达标约束的最早时间
        group_optimal_time = self._find_group_optimal_time_with_constraint(
            bmi_group_data, time_predictor, min_week, max_week
        )
        
        # 为该组的每个患者计算在该时间点的风险
        individual_results = []
        individual_risks = []
        individual_success_probs = []
        
        for _, patient in bmi_group_data.iterrows():
            risk_breakdown = self.calculate_total_risk(
                group_optimal_time, patient['孕妇BMI'], time_predictor
            )
            
            individual_results.append({
                '孕妇代码': patient.get('孕妇代码', 'Unknown'),
                'optimal_time_days': group_optimal_time,
                'optimal_time_weeks': group_optimal_time / 7,
                'total_risk': risk_breakdown['total_risk'],
                'success_probability': risk_breakdown['success_probability'],
                'detection_failure_risk': risk_breakdown['detection_failure_risk'],
                'delay_risk': risk_breakdown['delay_risk'],
                'satisfies_constraint': risk_breakdown['satisfies_constraint']
            })
            
            individual_risks.append(risk_breakdown['total_risk'])
            individual_success_probs.append(risk_breakdown['success_probability'])
        
        # 计算组统计结果
        group_optimal_risk = np.mean(individual_risks)
        
        # 检查组内稳妥达标约束：inf π(τ_g, b) ≥ p₀
        constraint_valid, min_probability = self.check_within_group_robust_constraint(
            bmi_group_data, group_optimal_time / 7, time_predictor
        )
        
        # 如果约束不满足，尝试调整时间
        if not constraint_valid:
            print(f"警告：组内稳妥达标约束不满足，尝试调整检测时间...")
            adjusted_time = self._adjust_time_for_constraint(
                bmi_group_data, time_predictor, group_optimal_time, min_week, max_week
            )
            
            if adjusted_time != group_optimal_time:
                group_optimal_time = adjusted_time
                # 重新计算所有指标
                individual_results = []
                individual_risks = []
                individual_success_probs = []
                
                for _, patient in bmi_group_data.iterrows():
                    risk_breakdown = self.calculate_total_risk(
                        group_optimal_time, patient['孕妇BMI'], time_predictor
                    )
                    
                    individual_results.append({
                        '孕妇代码': patient.get('孕妇代码', 'Unknown'),
                        'optimal_time_days': group_optimal_time,
                        'optimal_time_weeks': group_optimal_time / 7,
                        'total_risk': risk_breakdown['total_risk'],
                        'success_probability': risk_breakdown['success_probability'],
                        'detection_failure_risk': risk_breakdown['detection_failure_risk'],
                        'delay_risk': risk_breakdown['delay_risk'],
                        'satisfies_constraint': risk_breakdown['satisfies_constraint']
                    })
                    
                    individual_risks.append(risk_breakdown['total_risk'])
                    individual_success_probs.append(risk_breakdown['success_probability'])
                
                group_optimal_risk = np.mean(individual_risks)
                constraint_valid, min_probability = self.check_within_group_robust_constraint(
                    bmi_group_data, group_optimal_time / 7, time_predictor
                )
        
        # 计算组的综合风险分析
        group_risk_analysis = []
        for _, patient in bmi_group_data.iterrows():
            risk_breakdown = self.calculate_total_risk(
                group_optimal_time, patient['孕妇BMI'], time_predictor
            )
            group_risk_analysis.append(risk_breakdown)
        
        # 统计组层面的成功率
        success_rates = [r['success_probability'] for r in group_risk_analysis]
        constraint_satisfaction_rate = np.mean([r['satisfies_constraint'] for r in group_risk_analysis])
        
        return {
            'optimal_test_time_days': group_optimal_time,
            'optimal_test_time_weeks': group_optimal_time / 7,
            'minimal_expected_risk': group_optimal_risk,
            'optimization_success': True,
            'group_success_rate_mean': np.mean(success_rates),
            'group_success_rate_min': np.min(success_rates),
            'constraint_satisfaction_rate': constraint_satisfaction_rate,
            'within_group_robust_constraint_valid': constraint_valid,
            'within_group_min_probability': min_probability,
            'individual_results': individual_results,
            'sample_size': len(bmi_group_data),
            'detailed_analysis': {
                'total_risk_mean': np.mean([r['total_risk'] for r in group_risk_analysis]),
                'detection_failure_risk_mean': np.mean([r['detection_failure_risk'] for r in group_risk_analysis]),
                'delay_risk_mean': np.mean([r['delay_risk'] for r in group_risk_analysis]),
                'success_rate': np.mean(success_rates)
            }
        }
    
    def analyze_sensitivity_to_error(self, test_time: float, bmi_group_data: pd.DataFrame,
                                   time_predictor, error_range: List[float]) -> pd.DataFrame:
        """
        分析检测误差对结果的敏感性
        
        Parameters:
        - test_time: 检测时间
        - bmi_group_data: BMI组数据
        - time_predictor: 时间预测模型
        - error_range: 误差范围列表
        
        Returns:
        - sensitivity_df: 敏感性分析结果
        """
        results = []
        
        original_error = self.measurement_error_std
        
        for error_std in error_range:
            # 临时设置误差水平
            self.measurement_error_std = error_std
            
            # 计算该误差水平下的期望风险
            risk_analysis = self.calculate_expected_risk(test_time, bmi_group_data, time_predictor)
            
            results.append({
                'measurement_error_std': error_std,
                'expected_risk': risk_analysis['expected_risk'],
                'success_rate': risk_analysis['success_rate'],
                'constraint_satisfaction_rate': risk_analysis.get('constraint_satisfaction_rate', 0),
                'average_detection_failure_risk': risk_analysis.get('average_detection_failure_risk', 0),
                'average_delay_risk': risk_analysis.get('average_delay_risk', 0)
            })
        
        # 恢复原始误差设置
        self.measurement_error_std = original_error
        
        return pd.DataFrame(results)
    
    def _find_group_optimal_time_with_constraint(self, bmi_group_data: pd.DataFrame, 
                                               time_predictor, min_week: int, max_week: int) -> float:
        """
        找到满足组内稳妥达标约束的最优时间
        
        Parameters:
        - bmi_group_data: BMI组数据
        - time_predictor: 时间预测模型
        - min_week, max_week: 检测时间范围（周）
        
        Returns:
        - optimal_time: 最优检测时间（天）
        """
        import numpy as np
        
        # 为每个患者找到满足95%概率的最早时间
        individual_min_times = []
        
        for _, patient in bmi_group_data.iterrows():
            min_time = time_predictor.find_time_for_success_probability(
                patient['孕妇BMI'],
                target_prob=self.target_success_probability
            )
            
            if min_time is not None:
                individual_min_times.append(min_time)
        
        if not individual_min_times:
            # 如果所有患者都无法达到95%，使用最大时间
            return max_week * 7
        
        # 取所有患者中最大的最小时间，确保所有患者都能满足约束
        group_min_time = max(individual_min_times)
        
        # 确保在合理范围内
        group_min_time = max(group_min_time, min_week * 7)
        group_min_time = min(group_min_time, max_week * 7)
        
        return group_min_time
    
    def _adjust_time_for_constraint(self, bmi_group_data: pd.DataFrame, time_predictor,
                                  current_time: float, min_week: int, max_week: int) -> float:
        """
        调整检测时间以满足组内稳妥达标约束
        
        Parameters:
        - bmi_group_data: BMI组数据
        - time_predictor: 时间预测模型
        - current_time: 当前检测时间（天）
        - min_week, max_week: 检测时间范围（周）
        
        Returns:
        - adjusted_time: 调整后的检测时间（天）
        """
        import numpy as np
        
        # 在合理范围内搜索满足约束的时间
        search_times = np.linspace(current_time, max_week * 7, 20)
        
        for test_time in search_times:
            constraint_valid, min_prob = self.check_within_group_robust_constraint(
                bmi_group_data, test_time / 7, time_predictor
            )
            
            if constraint_valid:
                return test_time
        
        # 如果仍然无法满足约束，返回当前时间
        return current_time


if __name__ == "__main__":
    # 测试风险模型
    risk_model = RiskModel()
    
    # 测试时间风险计算
    print("时间风险测试:")
    for days in [80, 120, 200]:
        risk = risk_model.calculate_delay_risk(days)
        print(f"  {days}天 ({days/7:.1f}周): 风险分数 = {risk}")
    
    # 测试检测误差风险
    print("\n检测误差风险测试:")
    error_risk = risk_model.calculate_detection_error_risk(0.045, 120)
    for key, value in error_risk.items():
        print(f"  {key}: {value:.4f}")
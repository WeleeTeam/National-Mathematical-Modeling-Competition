"""
问题3风险模型
考虑多因素影响和达标比例的风险评估模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RiskModelProblem3:
    """问题3风险模型 - 考虑多因素影响和达标比例"""
    
    def __init__(self):
        # 概率约束参数
        self.target_success_probability = 0.95  # 95%成功概率要求
        self.measurement_error_std = 0.005       # Y染色体浓度测量标准误差
        self.individual_effect_std = 0.01        # 个体差异标准差
        
        # 约束条件参数
        self.bmi_min = 20.0      # BMI最小值
        self.bmi_max = 50.0      # BMI最大值
        self.detection_min_week = 10  # 检测时窗最小值 (10周)
        self.detection_max_week = 25  # 检测时窗最大值 (25周)
        self.detection_min_days = self.detection_min_week * 7
        self.detection_max_days = self.detection_max_week * 7
        
        # 多因素风险权重
        self.multifactor_weights = {
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
        
        # 延误诊断风险函数参数（考虑多因素影响）
        self.delay_risk_params = {
            'early_base': 0.1,      # 早期基础风险
            'early_slope': 0.02,    # 早期风险斜率  
            'medium_base': 0.5,     # 中期基础风险
            'medium_slope': 0.08,   # 中期风险斜率
            'late_base': 2.0,       # 晚期基础风险
            'late_slope': 0.2,      # 晚期风险斜率
            'multifactor_modifier': 1.2  # 多因素修正系数
        }
        
        # 检测失败惩罚系数（考虑多因素）
        self.detection_failure_penalty = 10.0
        self.multifactor_penalty_modifier = 1.5  # 多因素惩罚修正
    
    def calculate_multifactor_delay_risk(self, gestational_days: float, 
                                       multifactor_score: float) -> float:
        """
        计算考虑多因素影响的延误诊断风险
        
        Parameters:
        - gestational_days: 孕周天数
        - multifactor_score: 多因素综合评分
        
        Returns:
        - delay_risk: 延误诊断风险
        """
        weeks = gestational_days / 7
        
        # 基础延误风险
        if weeks <= 12:
            base_risk = (self.delay_risk_params['early_base'] + 
                        self.delay_risk_params['early_slope'] * weeks)
        elif weeks <= 20:
            base_risk = (self.delay_risk_params['medium_base'] + 
                        self.delay_risk_params['medium_slope'] * (weeks - 12))
        else:
            base_risk = (self.delay_risk_params['late_base'] + 
                        self.delay_risk_params['late_slope'] * (weeks - 20))
        
        # 多因素修正
        multifactor_modifier = 1 + (multifactor_score - 0.5) * self.delay_risk_params['multifactor_modifier']
        
        return base_risk * multifactor_modifier
    
    def calculate_multifactor_detection_failure_risk(self, success_probability: float,
                                                   multifactor_score: float) -> float:
        """
        计算考虑多因素影响的检测失败风险
        
        Parameters:
        - success_probability: 成功概率
        - multifactor_score: 多因素综合评分
        
        Returns:
        - detection_failure_risk: 检测失败风险
        """
        # 基础检测失败风险
        base_risk = (1 - success_probability) * self.detection_failure_penalty
        
        # 多因素修正
        multifactor_modifier = 1 + (multifactor_score - 0.5) * self.multifactor_penalty_modifier
        
        return base_risk * multifactor_modifier
    
    def calculate_multifactor_total_risk(self, gestational_days: float, bmi: float, 
                                       age: float, height: float, weight: float, 
                                       time_predictor, multifactor_score: float = None) -> Dict:
        """
        计算考虑多因素的总风险
        
        Parameters:
        - gestational_days: 孕周天数
        - bmi: 孕妇BMI
        - age: 孕妇年龄
        - height: 孕妇身高
        - weight: 孕妇体重
        - time_predictor: 时间预测模型
        - multifactor_score: 多因素综合评分（如果为None则计算）
        
        Returns:
        - risk_result: 风险计算结果
        """
        # 如果未提供多因素评分，则计算
        if multifactor_score is None:
            score_result = time_predictor.calculate_multifactor_score(bmi, age, height, weight)
            multifactor_score = score_result['total_score']
        
        # 预测成功概率
        success_probability = time_predictor.predict_达标概率(
            gestational_days, bmi, age, height, weight
        )
        
        # 计算各项风险
        delay_risk = self.calculate_multifactor_delay_risk(gestational_days, multifactor_score)
        detection_failure_risk = self.calculate_multifactor_detection_failure_risk(
            success_probability, multifactor_score
        )
        
        # 总风险
        total_risk = delay_risk + detection_failure_risk
        
        # 约束检查
        satisfies_constraint = success_probability >= self.target_success_probability
        
        return {
            'total_risk': total_risk,
            'delay_risk': delay_risk,
            'detection_failure_risk': detection_failure_risk,
            'success_probability': success_probability,
            'satisfies_constraint': satisfies_constraint,
            'multifactor_score': multifactor_score,
            'multifactor_risk_modifier': 1 + (multifactor_score - 0.5) * 0.5
        }
    
    def calculate_group_达标比例(self, group_data: pd.DataFrame, 
                                test_time_weeks: float, time_predictor) -> Dict:
        """
        计算组内达标比例
        
        Parameters:
        - group_data: 组内数据
        - test_time_weeks: 检测时间（周）
        - time_predictor: 时间预测模型
        
        Returns:
        - proportion_result: 达标比例结果
        """
        test_time_days = test_time_weeks * 7
        success_count = 0
        total_count = len(group_data)
        
        success_probabilities = []
        multifactor_scores = []
        
        for _, patient in group_data.iterrows():
            # 计算成功概率
            success_prob = time_predictor.predict_达标概率(
                test_time_days, patient['BMI'], patient['年龄'], 
                patient['身高'], patient['体重']
            )
            
            # 计算多因素评分
            score_result = time_predictor.calculate_multifactor_score(
                patient['BMI'], patient['年龄'], patient['身高'], patient['体重']
            )
            multifactor_score = score_result['total_score']
            
            success_probabilities.append(success_prob)
            multifactor_scores.append(multifactor_score)
            
            # 判断是否达标
            if success_prob >= self.target_success_probability:
                success_count += 1
        
        # 计算达标比例
        success_proportion = success_count / total_count if total_count > 0 else 0
        
        # 计算组内统计
        avg_success_prob = np.mean(success_probabilities)
        avg_multifactor_score = np.mean(multifactor_scores)
        min_success_prob = np.min(success_probabilities)
        
        return {
            'success_proportion': success_proportion,
            'success_count': success_count,
            'total_count': total_count,
            'avg_success_probability': avg_success_prob,
            'min_success_probability': min_success_prob,
            'avg_multifactor_score': avg_multifactor_score,
            'multifactor_score_std': np.std(multifactor_scores),
            'success_probability_std': np.std(success_probabilities)
        }
    
    def optimize_test_time_for_multifactor_group(self, group_data: pd.DataFrame, 
                                               time_predictor, min_week: int = 10, 
                                               max_week: int = 22) -> Dict:
        """
        为多因素组优化检测时间
        
        Parameters:
        - group_data: 组内数据
        - time_predictor: 时间预测模型
        - min_week: 最小检测周数
        - max_week: 最大检测周数
        
        Returns:
        - optimization_result: 优化结果
        """
        from scipy.optimize import minimize_scalar
        
        def group_risk_objective(test_time_weeks):
            test_time_days = test_time_weeks * 7
            total_risk = 0
            valid_patients = 0
            
            for _, patient in group_data.iterrows():
                try:
                    risk_result = self.calculate_multifactor_total_risk(
                        test_time_days, patient['BMI'], patient['年龄'], 
                        patient['身高'], patient['体重'], time_predictor
                    )
                    total_risk += risk_result['total_risk']
                    valid_patients += 1
                except:
                    continue
            
            return total_risk / valid_patients if valid_patients > 0 else float('inf')
        
        # 优化检测时间
        try:
            result = minimize_scalar(
                group_risk_objective,
                bounds=(min_week, max_week),
                method='bounded'
            )
            
            optimal_time_weeks = result.x
            minimal_risk = result.fun
            
        except:
            # 如果优化失败，使用组内平均时间
            optimal_times = []
            for _, patient in group_data.iterrows():
                optimal_time = time_predictor.solve_达标时间(
                    patient['BMI'], patient['年龄'], patient['身高'], patient['体重']
                )
                if optimal_time:
                    optimal_times.append(optimal_time / 7)
            
            optimal_time_weeks = np.mean(optimal_times) if optimal_times else (min_week + max_week) / 2
            minimal_risk = group_risk_objective(optimal_time_weeks)
        
        # 计算详细分析
        detailed_analysis = self.calculate_group_达标比例(group_data, optimal_time_weeks, time_predictor)
        
        # 计算组内稳妥达标约束
        within_group_robust_constraint_valid = detailed_analysis['min_success_probability'] >= self.target_success_probability
        
        return {
            'optimal_test_time_weeks': optimal_time_weeks,
            'optimal_test_time_days': optimal_time_weeks * 7,
            'minimal_expected_risk': minimal_risk,
            'detailed_analysis': detailed_analysis,
            'within_group_robust_constraint_valid': within_group_robust_constraint_valid,
            'within_group_min_probability': detailed_analysis['min_success_probability'],
            'group_success_rate_mean': detailed_analysis['avg_success_probability'],
            'constraint_satisfaction_rate': detailed_analysis['success_proportion'],
            'sample_size': len(group_data)
        }
    
    def analyze_multifactor_sensitivity_to_error(self, test_time: float, 
                                               group_data: pd.DataFrame,
                                               time_predictor, 
                                               error_range: List[float]) -> pd.DataFrame:
        """
        分析多因素组对检测误差的敏感性
        
        Parameters:
        - test_time: 检测时间（天）
        - group_data: 组内数据
        - time_predictor: 时间预测模型
        - error_range: 误差范围列表
        
        Returns:
        - sensitivity_results: 敏感性分析结果
        """
        results = []
        
        for error_std in error_range:
            # 临时修改测量误差
            original_error = time_predictor.measurement_error_std
            time_predictor.measurement_error_std = error_std
            
            # 计算组内风险
            group_risks = []
            success_rates = []
            
            for _, patient in group_data.iterrows():
                try:
                    risk_result = self.calculate_multifactor_total_risk(
                        test_time, patient['BMI'], patient['年龄'], 
                        patient['身高'], patient['体重'], time_predictor
                    )
                    group_risks.append(risk_result['total_risk'])
                    success_rates.append(risk_result['success_probability'])
                except:
                    continue
            
            # 恢复原始误差
            time_predictor.measurement_error_std = original_error
            
            if group_risks:
                results.append({
                    'measurement_error_std': error_std,
                    'expected_risk': np.mean(group_risks),
                    'risk_std': np.std(group_risks),
                    'success_rate': np.mean(success_rates),
                    'success_rate_std': np.std(success_rates)
                })
            else:
                print(f"警告: 误差 {error_std} 时没有有效的风险计算结果")
        
        if not results:
            print("警告: 敏感性分析没有返回任何结果")
            # 返回一个默认的空结果
            return pd.DataFrame({
                'measurement_error_std': error_range,
                'expected_risk': [0.0] * len(error_range),
                'risk_std': [0.0] * len(error_range),
                'success_rate': [0.0] * len(error_range),
                'success_rate_std': [0.0] * len(error_range)
            })
        
        return pd.DataFrame(results)
    
    def check_multifactor_constraints(self, groups: List[Dict], detection_times: List[float]) -> Dict:
        """
        检查多因素分组约束条件
        
        Parameters:
        - groups: 分组信息列表
        - detection_times: 检测时间列表
        
        Returns:
        - constraint_result: 约束检查结果
        """
        # 约束1：BMI分段约束
        bmi_groups = [(group['bmi_lower_bound'], group['bmi_upper_bound']) for group in groups]
        bmi_constraint_valid = self._check_bmi_segmentation_constraints(bmi_groups)
        
        # 约束2：检测时窗约束
        time_constraint_valid = self._check_detection_time_constraints(detection_times)
        
        # 约束3：多因素组内稳妥达标约束
        multifactor_constraint_valid = all(
            group.get('within_group_robust_constraint_valid', False) for group in groups
        )
        
        return {
            'bmi_segmentation_valid': bmi_constraint_valid,
            'detection_time_valid': time_constraint_valid,
            'multifactor_robust_constraint_valid': multifactor_constraint_valid,
            'overall_valid': bmi_constraint_valid and time_constraint_valid and multifactor_constraint_valid
        }
    
    def _check_bmi_segmentation_constraints(self, bmi_groups: List[Tuple[float, float]]) -> bool:
        """检查BMI分段约束"""
        if not bmi_groups:
            return False
        
        # 按边界排序
        bmi_groups.sort(key=lambda x: x[0])
        
        # 检查覆盖约束
        global_min = min(bmi_groups[0][0], self.bmi_min)
        global_max = max(bmi_groups[-1][1], self.bmi_max)
        
        if bmi_groups[0][0] > global_min or bmi_groups[-1][1] < global_max:
            return False
        
        # 检查相邻约束
        for i in range(len(bmi_groups) - 1):
            current_upper = bmi_groups[i][1]
            next_lower = bmi_groups[i + 1][0]
            
            if abs(current_upper - next_lower) > 1e-6:
                return False
        
        # 检查单调性
        boundaries = [bmi_groups[0][0]] + [group[1] for group in bmi_groups]
        for i in range(len(boundaries) - 1):
            if boundaries[i] >= boundaries[i + 1]:
                return False
        
        return True
    
    def _check_detection_time_constraints(self, detection_times: List[float]) -> bool:
        """检查检测时窗约束"""
        if not detection_times:
            return False
        
        return all(
            self.detection_min_week <= time <= self.detection_max_week 
            for time in detection_times
        )


if __name__ == "__main__":
    # 测试问题3风险模型
    print("问题3风险模型测试")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'BMI': [25.0, 32.0, 40.0],
        '年龄': [28, 30, 35],
        '身高': [160, 165, 170],
        '体重': [60, 80, 100]
    })
    
    risk_model = RiskModelProblem3()
    print(f"风险模型初始化完成")
    print(f"多因素权重: {risk_model.multifactor_weights}")
    
    # 测试多因素风险计算
    print("\n多因素风险计算测试:")
    for _, patient in test_data.iterrows():
        risk = risk_model.calculate_multifactor_delay_risk(100, 0.5)
        print(f"BMI {patient['BMI']}: 延误风险 = {risk:.4f}")
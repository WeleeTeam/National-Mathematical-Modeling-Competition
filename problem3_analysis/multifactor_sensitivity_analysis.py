"""
多因素敏感性分析模块
提供全面的多因素敏感性分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultifactorSensitivityAnalyzer:
    """多因素敏感性分析器"""
    
    def __init__(self, risk_model, time_predictor):
        self.risk_model = risk_model
        self.time_predictor = time_predictor
        self.results = {}
        
    def comprehensive_sensitivity_analysis(self, groups: Dict, optimization_results: Dict) -> Dict:
        """
        综合多因素敏感性分析
        
        Parameters:
        - groups: 分组信息
        - optimization_results: 优化结果
        
        Returns:
        - comprehensive_results: 综合敏感性分析结果
        """
        print("\n=== 综合多因素敏感性分析 ===")
        
        comprehensive_results = {}
        
        # 1. 检测误差敏感性分析
        print("\n1. 检测误差敏感性分析...")
        error_sensitivity = self.analyze_measurement_error_sensitivity(groups, optimization_results)
        comprehensive_results['error_sensitivity'] = error_sensitivity
        
        # 2. 孕周敏感性分析
        print("\n2. 孕周敏感性分析...")
        gestational_sensitivity = self.analyze_gestational_week_sensitivity(groups, optimization_results)
        comprehensive_results['gestational_sensitivity'] = gestational_sensitivity
        
        # 3. 成功概率约束敏感性分析
        print("\n3. 成功概率约束敏感性分析...")
        constraint_sensitivity = self.analyze_success_probability_sensitivity(groups, optimization_results)
        comprehensive_results['constraint_sensitivity'] = constraint_sensitivity
        
        # 4. 多因素权重敏感性分析
        print("\n4. 多因素权重敏感性分析...")
        weight_sensitivity = self.analyze_multifactor_weight_sensitivity(groups, optimization_results)
        comprehensive_results['weight_sensitivity'] = weight_sensitivity
        
        # 5. 样本量敏感性分析
        print("\n5. 样本量敏感性分析...")
        sample_sensitivity = self.analyze_sample_size_sensitivity(groups, optimization_results)
        comprehensive_results['sample_sensitivity'] = sample_sensitivity
        
        # 6. 个体差异敏感性分析
        print("\n6. 个体差异敏感性分析...")
        individual_sensitivity = self.analyze_individual_variation_sensitivity(groups, optimization_results)
        comprehensive_results['individual_sensitivity'] = individual_sensitivity
        
        self.results = comprehensive_results
        return comprehensive_results
    
    def analyze_measurement_error_sensitivity(self, groups: Dict, optimization_results: Dict) -> Dict:
        """分析检测误差敏感性"""
        error_ranges = {
            'low': [0.001, 0.002, 0.003],
            'medium': [0.004, 0.005, 0.006],
            'high': [0.008, 0.010, 0.015, 0.020]
        }
        
        results = {}
        
        for group_name, group_info in groups.items():
            print(f"  分析 {group_name} 的检测误差敏感性...")
            
            optimal_time_days = optimization_results[group_name]['optimal_test_time_days']
            group_data = self._prepare_group_data(group_info)
            
            group_results = {}
            
            for error_level, error_values in error_ranges.items():
                error_results = []
                
                for error_std in error_values:
                    # 临时修改测量误差
                    original_error = self.time_predictor.measurement_error_std
                    self.time_predictor.measurement_error_std = error_std
                    
                    # 计算风险
                    risks = []
                    success_rates = []
                    
                    for _, patient in group_data.iterrows():
                        try:
                            risk_result = self.risk_model.calculate_multifactor_total_risk(
                                optimal_time_days, patient['BMI'], patient['年龄'],
                                patient['身高'], patient['体重'], self.time_predictor
                            )
                            risks.append(risk_result['total_risk'])
                            success_rates.append(risk_result['success_probability'])
                        except Exception as e:
                            continue
                    
                    # 恢复原始误差
                    self.time_predictor.measurement_error_std = original_error
                    
                    if risks:
                        error_results.append({
                            'error_std': error_std,
                            'expected_risk': np.mean(risks),
                            'risk_std': np.std(risks),
                            'success_rate': np.mean(success_rates),
                            'success_rate_std': np.std(success_rates)
                        })
                
                group_results[error_level] = pd.DataFrame(error_results)
            
            results[group_name] = group_results
        
        return results
    
    def analyze_gestational_week_sensitivity(self, groups: Dict, optimization_results: Dict) -> Dict:
        """分析孕周敏感性"""
        week_offsets = [-3, -2, -1, 0, 1, 2, 3]  # 周数偏移
        
        results = {}
        
        for group_name, group_info in groups.items():
            print(f"  分析 {group_name} 的孕周敏感性...")
            
            optimal_time_days = optimization_results[group_name]['optimal_test_time_days']
            group_data = self._prepare_group_data(group_info)
            
            week_results = []
            
            for week_offset in week_offsets:
                test_time_days = optimal_time_days + (week_offset * 7)
                
                # 确保在合理范围内
                test_time_days = max(70, min(175, test_time_days))  # 10-25周
                
                risks = []
                success_rates = []
                
                for _, patient in group_data.iterrows():
                    try:
                        risk_result = self.risk_model.calculate_multifactor_total_risk(
                            test_time_days, patient['BMI'], patient['年龄'],
                            patient['身高'], patient['体重'], self.time_predictor
                        )
                        risks.append(risk_result['total_risk'])
                        success_rates.append(risk_result['success_probability'])
                    except Exception as e:
                        continue
                
                if risks:
                    week_results.append({
                        'week_offset': week_offset,
                        'test_time_days': test_time_days,
                        'test_time_weeks': test_time_days / 7,
                        'expected_risk': np.mean(risks),
                        'risk_std': np.std(risks),
                        'success_rate': np.mean(success_rates),
                        'success_rate_std': np.std(success_rates)
                    })
            
            results[group_name] = pd.DataFrame(week_results)
        
        return results
    
    def analyze_success_probability_sensitivity(self, groups: Dict, optimization_results: Dict) -> Dict:
        """分析成功概率约束敏感性"""
        success_probabilities = [0.85, 0.90, 0.95, 0.98, 0.99]
        
        results = {}
        
        for group_name, group_info in groups.items():
            print(f"  分析 {group_name} 的成功概率约束敏感性...")
            
            group_data = self._prepare_group_data(group_info)
            
            prob_results = []
            
            for target_prob in success_probabilities:
                # 临时修改目标成功概率
                original_prob = self.risk_model.target_success_probability
                self.risk_model.target_success_probability = target_prob
                
                # 重新优化检测时间
                try:
                    opt_result = self.risk_model.optimize_test_time_for_multifactor_group(
                        group_data, self.time_predictor, min_week=10, max_week=25
                    )
                    
                    prob_results.append({
                        'target_probability': target_prob,
                        'optimal_time_days': opt_result['optimal_test_time_days'],
                        'optimal_time_weeks': opt_result['optimal_test_time_weeks'],
                        'minimal_risk': opt_result['minimal_expected_risk'],
                        'actual_success_rate': opt_result['detailed_analysis']['success_proportion']
                    })
                except Exception as e:
                    print(f"    警告: 成功概率 {target_prob} 优化失败: {e}")
                    continue
                
                # 恢复原始概率
                self.risk_model.target_success_probability = original_prob
            
            results[group_name] = pd.DataFrame(prob_results)
        
        return results
    
    def analyze_multifactor_weight_sensitivity(self, groups: Dict, optimization_results: Dict) -> Dict:
        """分析多因素权重敏感性"""
        weight_variations = {
            'bmi': [0.15, 0.20, 0.25, 0.30, 0.35],
            'age': [0.10, 0.15, 0.20, 0.25, 0.30],
            'height': [0.05, 0.10, 0.15, 0.20, 0.25],
            'weight': [0.10, 0.15, 0.20, 0.25, 0.30]
        }
        
        results = {}
        
        for group_name, group_info in groups.items():
            print(f"  分析 {group_name} 的多因素权重敏感性...")
            
            group_data = self._prepare_group_data(group_info)
            group_results = {}
            
            for factor, weights in weight_variations.items():
                factor_results = []
                
                for weight in weights:
                    # 临时修改权重
                    original_weights = self.risk_model.multifactor_weights.copy()
                    self.risk_model.multifactor_weights[factor] = weight
                    
                    # 重新计算风险
                    try:
                        optimal_time_days = optimization_results[group_name]['optimal_test_time_days']
                        
                        risks = []
                        for _, patient in group_data.iterrows():
                            risk_result = self.risk_model.calculate_multifactor_total_risk(
                                optimal_time_days, patient['BMI'], patient['年龄'],
                                patient['身高'], patient['体重'], self.time_predictor
                            )
                            risks.append(risk_result['total_risk'])
                        
                        if risks:
                            factor_results.append({
                                'weight': weight,
                                'expected_risk': np.mean(risks),
                                'risk_std': np.std(risks)
                            })
                    except Exception as e:
                        continue
                    
                    # 恢复原始权重
                    self.risk_model.multifactor_weights = original_weights
                
                group_results[factor] = pd.DataFrame(factor_results)
            
            results[group_name] = group_results
        
        return results
    
    def analyze_sample_size_sensitivity(self, groups: Dict, optimization_results: Dict) -> Dict:
        """分析样本量敏感性（Bootstrap重采样）"""
        bootstrap_samples = [50, 75, 100, 125, 150]  # 样本量
        n_bootstrap = 20  # Bootstrap次数
        
        results = {}
        
        for group_name, group_info in groups.items():
            print(f"  分析 {group_name} 的样本量敏感性...")
            
            group_data = self._prepare_group_data(group_info)
            group_results = []
            
            for sample_size in bootstrap_samples:
                if len(group_data) < sample_size:
                    continue
                
                bootstrap_risks = []
                bootstrap_times = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap重采样
                    bootstrap_data = group_data.sample(n=sample_size, replace=True)
                    
                    try:
                        # 重新优化
                        opt_result = self.risk_model.optimize_test_time_for_multifactor_group(
                            bootstrap_data, self.time_predictor, min_week=10, max_week=25
                        )
                        
                        bootstrap_risks.append(opt_result['minimal_expected_risk'])
                        bootstrap_times.append(opt_result['optimal_test_time_weeks'])
                    except Exception as e:
                        continue
                
                if bootstrap_risks:
                    group_results.append({
                        'sample_size': sample_size,
                        'mean_risk': np.mean(bootstrap_risks),
                        'risk_std': np.std(bootstrap_risks),
                        'mean_time': np.mean(bootstrap_times),
                        'time_std': np.std(bootstrap_times),
                        'n_successful': len(bootstrap_risks)
                    })
            
            results[group_name] = pd.DataFrame(group_results)
        
        return results
    
    def analyze_individual_variation_sensitivity(self, groups: Dict, optimization_results: Dict) -> Dict:
        """分析个体差异敏感性"""
        variation_levels = [0.5, 0.8, 1.0, 1.2, 1.5]  # 个体差异倍数
        
        results = {}
        
        for group_name, group_info in groups.items():
            print(f"  分析 {group_name} 的个体差异敏感性...")
            
            group_data = self._prepare_group_data(group_info)
            group_results = []
            
            for variation_factor in variation_levels:
                # 临时修改个体差异标准差
                original_std = self.risk_model.individual_effect_std
                self.risk_model.individual_effect_std = original_std * variation_factor
                
                risks = []
                success_rates = []
                
                optimal_time_days = optimization_results[group_name]['optimal_test_time_days']
                
                for _, patient in group_data.iterrows():
                    try:
                        risk_result = self.risk_model.calculate_multifactor_total_risk(
                            optimal_time_days, patient['BMI'], patient['年龄'],
                            patient['身高'], patient['体重'], self.time_predictor
                        )
                        risks.append(risk_result['total_risk'])
                        success_rates.append(risk_result['success_probability'])
                    except Exception as e:
                        continue
                
                if risks:
                    group_results.append({
                        'variation_factor': variation_factor,
                        'individual_std': self.risk_model.individual_effect_std,
                        'expected_risk': np.mean(risks),
                        'risk_std': np.std(risks),
                        'success_rate': np.mean(success_rates),
                        'success_rate_std': np.std(success_rates)
                    })
                
                # 恢复原始标准差
                self.risk_model.individual_effect_std = original_std
            
            results[group_name] = pd.DataFrame(group_results)
        
        return results
    
    def _prepare_group_data(self, group_info: Dict) -> pd.DataFrame:
        """准备组数据"""
        # group_info['group_data'] 应该已经包含了正确的患者数据
        group_patient_df = group_info['group_data'].copy()
        
        # 确保列名正确
        if '孕妇BMI' in group_patient_df.columns:
            group_patient_df = group_patient_df.rename(columns={'孕妇BMI': 'BMI'})
        
        return group_patient_df
    
    def generate_sensitivity_report(self, results: Dict) -> str:
        """生成敏感性分析报告"""
        report = []
        report.append("=" * 80)
        report.append("多因素敏感性分析报告")
        report.append("=" * 80)
        
        for analysis_type, analysis_results in results.items():
            report.append(f"\n## {analysis_type.replace('_', ' ').title()}")
            report.append("-" * 50)
            
            if isinstance(analysis_results, dict):
                for group_name, group_results in analysis_results.items():
                    report.append(f"\n### {group_name}")
                    
                    if isinstance(group_results, dict):
                        for factor, factor_data in group_results.items():
                            if isinstance(factor_data, pd.DataFrame) and not factor_data.empty:
                                report.append(f"\n{factor}:")
                                report.append(factor_data.to_string(index=False))
                    elif isinstance(group_results, pd.DataFrame) and not group_results.empty:
                        report.append(group_results.to_string(index=False))
        
        return "\n".join(report)
    
    def create_sensitivity_visualizations(self, results: Dict, save_path: str = "problem3_results/figures/"):
        """创建敏感性分析可视化图表"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 检测误差敏感性图
        self._plot_error_sensitivity(results.get('error_sensitivity', {}), save_path)
        
        # 2. 孕周敏感性图
        self._plot_gestational_sensitivity(results.get('gestational_sensitivity', {}), save_path)
        
        # 3. 成功概率约束敏感性图
        self._plot_constraint_sensitivity(results.get('constraint_sensitivity', {}), save_path)
        
        # 4. 多因素权重敏感性图
        self._plot_weight_sensitivity(results.get('weight_sensitivity', {}), save_path)
        
        print(f"敏感性分析图表已保存到: {save_path}")
    
    def _plot_error_sensitivity(self, error_results: Dict, save_path: str):
        """绘制检测误差敏感性图"""
        if not error_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('检测误差敏感性分析', fontsize=16, fontweight='bold')
        
        for i, (group_name, group_data) in enumerate(error_results.items()):
            if i >= 4:  # 最多显示4个组
                break
            
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            for error_level, data in group_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    ax.plot(data['error_std'], data['expected_risk'], 
                           marker='o', label=f'{error_level}误差', linewidth=2)
            
            ax.set_title(f'{group_name}', fontweight='bold')
            ax.set_xlabel('测量误差标准差')
            ax.set_ylabel('期望风险')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/error_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gestational_sensitivity(self, gestational_results: Dict, save_path: str):
        """绘制孕周敏感性图"""
        if not gestational_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('孕周敏感性分析', fontsize=16, fontweight='bold')
        
        for i, (group_name, data) in enumerate(gestational_results.items()):
            if i >= 4:
                break
            
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if isinstance(data, pd.DataFrame) and not data.empty:
                ax.plot(data['week_offset'], data['expected_risk'], 
                       marker='o', linewidth=2, markersize=8)
                ax.set_title(f'{group_name}', fontweight='bold')
                ax.set_xlabel('孕周偏移（周）')
                ax.set_ylabel('期望风险')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/gestational_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_constraint_sensitivity(self, constraint_results: Dict, save_path: str):
        """绘制约束敏感性图"""
        if not constraint_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('成功概率约束敏感性分析', fontsize=16, fontweight='bold')
        
        for i, (group_name, data) in enumerate(constraint_results.items()):
            if i >= 4:
                break
            
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if isinstance(data, pd.DataFrame) and not data.empty:
                ax.plot(data['target_probability'], data['optimal_time_weeks'], 
                       marker='o', linewidth=2, markersize=8)
                ax.set_title(f'{group_name}', fontweight='bold')
                ax.set_xlabel('目标成功概率')
                ax.set_ylabel('最优检测时间（周）')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/constraint_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_weight_sensitivity(self, weight_results: Dict, save_path: str):
        """绘制权重敏感性图"""
        if not weight_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('多因素权重敏感性分析', fontsize=16, fontweight='bold')
        
        for i, (group_name, group_data) in enumerate(weight_results.items()):
            if i >= 4:
                break
            
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            for factor, data in group_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    ax.plot(data['weight'], data['expected_risk'], 
                           marker='o', label=factor, linewidth=2)
            
            ax.set_title(f'{group_name}', fontweight='bold')
            ax.set_xlabel('权重值')
            ax.set_ylabel('期望风险')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/weight_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
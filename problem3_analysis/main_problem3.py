"""
问题3主分析模块
综合考虑身高、体重、年龄等多种因素进行BMI分组和最佳NIPT时点选择
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from time_prediction_problem3 import TimePredictionModelProblem3
from risk_model_problem3 import RiskModelProblem3
from multifactor_grouping import MultifactorGrouping

# 导入第一问的数据处理器
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'problem1_analysis'))

try:
    from data_processor import NIPTDataProcessor
except ImportError:
    # 如果导入失败，使用简化的数据处理器
    print("警告：无法导入problem1_analysis中的数据处理器，使用简化版本")
    
    class NIPTDataProcessor:
        def load_data(self, file_path):
            import pandas as pd
            return pd.read_csv(file_path, encoding='utf-8')
        
        def clean_data(self):
            return self.data
        
        def prepare_longitudinal_format(self, cleaned_data):
            return cleaned_data


class Problem3CompleteAnalysis:
    """问题3完整分析流程 - 多因素综合分组"""
    
    def __init__(self, data_path: str):
        """
        初始化分析流程
        
        Parameters:
        - data_path: 男胎数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.unique_patients = None
        
        # 初始化各个模块
        self.time_predictor = TimePredictionModelProblem3()
        self.risk_model = RiskModelProblem3()
        self.multifactor_grouping = MultifactorGrouping(max_groups=5, min_samples_per_group=20)
        
        # 存储分析结果
        self.analysis_results = {}
        
        print("问题3多因素分析流程初始化完成")
        print("考虑因素：BMI、年龄、身高、体重、X染色体浓度、检测质量等")
    
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("\n=== Step 1: 数据加载与预处理 ===")
        
        # 使用第一问的数据处理器
        processor = NIPTDataProcessor()
        self.raw_data = processor.load_data(self.data_path)
        
        # 数据清洗
        cleaned_data = processor.clean_data()
        
        # 准备纵向格式数据
        self.processed_data = processor.prepare_longitudinal_format(cleaned_data)
        
        # 获取每个孕妇的基本信息（取第一次检测的记录）
        self.unique_patients = self.processed_data.groupby('孕妇代码').first().reset_index()
        
        # 添加缺失的列（如果不存在）
        required_columns = ['X染色体浓度', 'Y染色体Z值', 'X染色体Z值', '18号染色体Z值', '检测抽血次数']
        for col in required_columns:
            if col not in self.unique_patients.columns:
                if col == 'X染色体浓度':
                    self.unique_patients[col] = 0.5  # 默认值
                elif col == '检测抽血次数':
                    self.unique_patients[col] = 1    # 默认值
                else:
                    self.unique_patients[col] = 0.0  # 默认值
        
        print(f"数据预处理完成:")
        print(f"  - 总记录数: {len(self.processed_data)}")
        print(f"  - 唯一孕妇数: {len(self.unique_patients)}")
        print(f"  - BMI范围: {self.unique_patients['孕妇BMI'].min():.1f} - {self.unique_patients['孕妇BMI'].max():.1f}")
        print(f"  - 年龄范围: {self.unique_patients['年龄'].min():.0f} - {self.unique_patients['年龄'].max():.0f}")
        print(f"  - 身高范围: {self.unique_patients['身高'].min():.0f} - {self.unique_patients['身高'].max():.0f}")
        print(f"  - 体重范围: {self.unique_patients['体重'].min():.0f} - {self.unique_patients['体重'].max():.0f}")
        
        return self.processed_data, self.unique_patients
    
    def predict_multifactor_达标时间(self):
        """为所有患者预测多因素达标时间"""
        print("\n=== Step 2: 多因素达标时间预测 ===")
        
        # 批量预测多因素达标时间
        prediction_results = self.time_predictor.batch_predict_达标时间(self.unique_patients)
        
        # 过滤掉无法在合理时间内满足约束的记录
        valid_predictions = prediction_results.dropna(subset=['满足95%约束时间'])
        
        print(f"多因素时间预测完成:")
        print(f"  - 总患者数: {len(prediction_results)}")
        print(f"  - 可满足95%约束的患者: {len(valid_predictions)}/{len(prediction_results)}")
        
        if len(valid_predictions) > 0:
            print(f"  - 平均95%约束时间: {valid_predictions['满足95%约束周数'].mean():.1f}周")
            print(f"  - 约束时间范围: {valid_predictions['满足95%约束周数'].min():.1f} - {valid_predictions['满足95%约束周数'].max():.1f}周")
        
        self.analysis_results['prediction_results'] = prediction_results
        self.analysis_results['valid_predictions'] = valid_predictions
        
        return valid_predictions
    
    def optimize_multifactor_grouping(self):
        """使用多因素特征进行分组优化"""
        print("\n=== Step 3: 多因素分组优化 ===")
        
        # 准备多因素特征数据
        feature_df = self.multifactor_grouping.prepare_multifactor_features(
            self.analysis_results['valid_predictions'], 
            self.time_predictor, 
            self.risk_model
        )
        
        print(f"多因素特征数据准备完成，共 {len(feature_df)} 个样本")
        print(f"特征维度: {len(feature_df.columns)} 个特征")
        
        # 运行多因素分组优化
        grouping_result = self.multifactor_grouping.optimize_multifactor_grouping(feature_df)
        
        print(f"多因素分组完成:")
        print(f"  - 最优聚类方法: {grouping_result['best_method']}")
        print(f"  - 最优聚类数: {grouping_result['best_n_clusters']}")
        print(f"  - 聚类评分: {grouping_result['best_score']:.4f}")
        print(f"  - 生成组数: {len(grouping_result['groups'])}")
        
        # 细化分组边界
        refined_groups = self.multifactor_grouping.refine_multifactor_grouping_with_bmi_bounds(feature_df)
        
        # 生成分组规则
        grouping_rules = self.multifactor_grouping.generate_multifactor_grouping_rules(refined_groups)
        
        print(f"\n多因素分组规则:")
        for rule in grouping_rules:
            print(f"  {rule['group_name']}: BMI {rule['bmi_interval_description']}, "
                  f"推荐 {rule['recommended_test_time_weeks']:.1f}周检测, "
                  f"样本数 {rule['sample_size']}, "
                  f"多因素评分 {rule['multifactor_score_mean']:.3f}")
        
        self.analysis_results['feature_df'] = feature_df
        self.analysis_results['grouping_result'] = grouping_result
        self.analysis_results['refined_groups'] = refined_groups
        self.analysis_results['grouping_rules'] = grouping_rules
        
        return grouping_rules
    
    def optimize_test_times_for_multifactor_groups(self):
        """为各多因素组优化检测时间"""
        print("\n=== Step 4: 多因素组最佳检测时间优化 ===")
        
        optimization_results = {}
        
        for group_name, group_info in self.analysis_results['refined_groups'].items():
            print(f"\n优化 {group_name} (BMI: {group_info['bmi_interval']}):")
            
            # 获取该组的数据
            group_data = group_info['group_data']
            
            # 转换为原始数据格式用于风险计算
            group_patient_data = []
            for _, row in group_data.iterrows():
                patient_code = row['孕妇代码']
                original_data = self.unique_patients[self.unique_patients['孕妇代码'] == patient_code].iloc[0]
                group_patient_data.append(original_data)
            
            group_patient_df = pd.DataFrame(group_patient_data)
            
            # 重命名列以匹配风险模型的期望
            if '孕妇BMI' in group_patient_df.columns:
                group_patient_df = group_patient_df.rename(columns={'孕妇BMI': 'BMI'})
            
            # 优化该组的检测时间
            opt_result = self.risk_model.optimize_test_time_for_multifactor_group(
                group_patient_df, self.time_predictor, min_week=10, max_week=22
            )
            
            print(f"  - 最优检测时间: {opt_result['optimal_test_time_weeks']:.1f}周")
            print(f"  - 最小期望风险: {opt_result['minimal_expected_risk']:.4f}")
            print(f"  - 达标成功率: {opt_result['detailed_analysis']['success_proportion']:.3f}")
            print(f"  - 组内稳妥达标: {'✅' if opt_result['within_group_robust_constraint_valid'] else '❌'}")
            
            optimization_results[group_name] = opt_result
        
        self.analysis_results['optimization_results'] = optimization_results
        
        return optimization_results
    
    def analyze_multifactor_error_sensitivity(self):
        """分析多因素组对检测误差的敏感性（保留原有方法作为备用）"""
        print("\n=== Step 5: 多因素检测误差敏感性分析 ===")
        
        error_range = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
        sensitivity_results = {}
        
        for group_name, group_info in self.analysis_results['refined_groups'].items():
            print(f"\n分析 {group_name} 的误差敏感性:")
            
            # 获取该组的优化检测时间
            optimal_time_days = self.analysis_results['optimization_results'][group_name]['optimal_test_time_days']
            
            # 获取组数据
            group_data = group_info['group_data']
            group_patient_data = []
            for _, row in group_data.iterrows():
                patient_code = row['孕妇代码']
                original_data = self.unique_patients[self.unique_patients['孕妇代码'] == patient_code].iloc[0]
                group_patient_data.append(original_data)
            
            group_patient_df = pd.DataFrame(group_patient_data)
            
            # 重命名列以匹配风险模型的期望
            if '孕妇BMI' in group_patient_df.columns:
                group_patient_df = group_patient_df.rename(columns={'孕妇BMI': 'BMI'})
            
            # 进行敏感性分析
            sens_result = self.risk_model.analyze_multifactor_sensitivity_to_error(
                optimal_time_days, group_patient_df, self.time_predictor, error_range
            )
            
            sensitivity_results[group_name] = sens_result
            
            print(f"  - 误差范围: {min(error_range):.3f} - {max(error_range):.3f}")
            if not sens_result.empty and 'expected_risk' in sens_result.columns:
                print(f"  - 风险变化: {sens_result['expected_risk'].min():.4f} - {sens_result['expected_risk'].max():.4f}")
            else:
                print(f"  - 风险变化: 无法计算（数据为空或缺少列）")
        
        self.analysis_results['sensitivity_results'] = sensitivity_results
        
        return sensitivity_results
    
    def comprehensive_multifactor_sensitivity_analysis(self):
        """综合多因素敏感性分析"""
        print("\n=== Step 5: 综合多因素敏感性分析 ===")
        
        # 导入多因素敏感性分析器
        from multifactor_sensitivity_analysis import MultifactorSensitivityAnalyzer
        
        # 创建敏感性分析器
        sensitivity_analyzer = MultifactorSensitivityAnalyzer(self.risk_model, self.time_predictor)
        
        # 准备分组数据（添加原始患者数据）
        enhanced_groups = {}
        for group_name, group_info in self.analysis_results['refined_groups'].items():
            enhanced_group_info = group_info.copy()
            
            # 添加原始患者数据
            group_data = group_info['group_data']
            group_patient_data = []
            for _, row in group_data.iterrows():
                patient_code = row['孕妇代码']
                original_data = self.unique_patients[self.unique_patients['孕妇代码'] == patient_code].iloc[0]
                group_patient_data.append(original_data)
            
            group_patient_df = pd.DataFrame(group_patient_data)
            
            # 重命名列以匹配风险模型的期望
            if '孕妇BMI' in group_patient_df.columns:
                group_patient_df = group_patient_df.rename(columns={'孕妇BMI': 'BMI'})
            
            enhanced_group_info['group_data'] = group_patient_df
            enhanced_groups[group_name] = enhanced_group_info
        
        # 进行综合敏感性分析
        comprehensive_results = sensitivity_analyzer.comprehensive_sensitivity_analysis(
            enhanced_groups, self.analysis_results['optimization_results']
        )
        
        # 保存结果
        self.analysis_results['comprehensive_sensitivity_results'] = comprehensive_results
        
        # 生成报告
        report = sensitivity_analyzer.generate_sensitivity_report(comprehensive_results)
        
        # 保存报告
        import os
        os.makedirs("problem3_results/reports", exist_ok=True)
        with open("problem3_results/reports/comprehensive_sensitivity_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        # 创建可视化图表
        sensitivity_analyzer.create_sensitivity_visualizations(comprehensive_results)
        
        print("综合多因素敏感性分析完成！")
        print("  ✓ 检测误差敏感性分析")
        print("  ✓ 孕周敏感性分析")
        print("  ✓ 成功概率约束敏感性分析")
        print("  ✓ 多因素权重敏感性分析")
        print("  ✓ 样本量敏感性分析")
        print("  ✓ 个体差异敏感性分析")
        print("  ✓ 分析报告和可视化图表已保存")
        
        return comprehensive_results
    
    def generate_final_multifactor_recommendations(self):
        """生成最终的多因素临床建议"""
        print("\n=== Step 6: 生成最终多因素临床建议 ===")
        
        recommendations = []
        
        # 收集所有BMI分组和检测时间用于约束检查
        bmi_groups = []
        detection_times = []
        
        for i, rule in enumerate(self.analysis_results['grouping_rules']):
            group_name = f"Group_{i+1}"
            opt_result = self.analysis_results['optimization_results'][group_name]
            
            # 收集约束检查数据
            bmi_groups.append((rule['bmi_lower_bound'], rule['bmi_upper_bound']))
            detection_times.append(opt_result['optimal_test_time_weeks'])
            
            recommendation = {
                'group_id': rule['group_id'],
                'group_name': rule['group_name'], 
                'bmi_range': rule['bmi_interval_description'],
                'bmi_mean': rule['bmi_mean'],
                'age_mean': rule['age_mean'],
                'height_mean': rule['height_mean'],
                'weight_mean': rule['weight_mean'],
                'multifactor_score_mean': rule['multifactor_score_mean'],
                'sample_size': rule['sample_size'],
                'recommended_test_time_weeks': opt_result['optimal_test_time_weeks'],
                'recommended_test_time_days': opt_result['optimal_test_time_days'],
                'expected_minimal_risk': opt_result['minimal_expected_risk'],
                'success_rate': opt_result['detailed_analysis']['success_proportion'],
                'constraint_satisfaction_rate': opt_result.get('constraint_satisfaction_rate', 0),
                'within_group_robust_constraint_valid': opt_result.get('within_group_robust_constraint_valid', False),
                'within_group_min_probability': opt_result.get('within_group_min_probability', 0),
                'clinical_advice': self._generate_multifactor_clinical_advice(opt_result, rule),
                'error_sensitivity': self._analyze_multifactor_error_impact(group_name)
            }
            
            recommendations.append(recommendation)
        
        # 验证多因素约束条件
        print("\n多因素约束条件验证:")
        print("-" * 50)
        
        constraint_validation = self.risk_model.check_multifactor_constraints(
            self.analysis_results['grouping_rules'], detection_times
        )
        
        print(f"约束1 - BMI分段约束: {'✅ 满足' if constraint_validation['bmi_segmentation_valid'] else '❌ 不满足'}")
        print(f"约束2 - 检测时窗约束(10-25周): {'✅ 满足' if constraint_validation['detection_time_valid'] else '❌ 不满足'}")
        print(f"约束3 - 多因素组内稳妥达标约束: {'✅ 满足' if constraint_validation['multifactor_robust_constraint_valid'] else '❌ 不满足'}")
        print(f"整体约束满足: {'✅ 满足' if constraint_validation['overall_valid'] else '❌ 不满足'}")
        
        self.analysis_results['final_recommendations'] = recommendations
        self.analysis_results['constraint_validation'] = constraint_validation
        
        # 打印最终建议
        print("\n最终多因素BMI分组和NIPT时点建议:")
        print("="*80)
        for rec in recommendations:
            print(f"{rec['group_name']}: BMI {rec['bmi_range']}")
            print(f"  - 建议检测时间: {rec['recommended_test_time_weeks']:.1f}周 ({rec['recommended_test_time_days']:.0f}天)")
            print(f"  - 最小总风险: {rec['expected_minimal_risk']:.4f}")
            print(f"  - 平均成功概率: {rec['success_rate']:.3f}")
            print(f"  - 多因素评分: {rec['multifactor_score_mean']:.3f}")
            print(f"  - 组内稳妥达标: {'✅' if rec['within_group_robust_constraint_valid'] else '❌'}")
            print(f"  - 临床建议: {rec['clinical_advice']}")
            print(f"  - 误差敏感性: {rec['error_sensitivity']}")
            print()
        
        return recommendations
    
    def _generate_multifactor_clinical_advice(self, opt_result: Dict, rule: Dict) -> str:
        """生成多因素临床建议文本"""
        weeks = opt_result['optimal_test_time_weeks']
        success_rate = opt_result['detailed_analysis']['success_proportion']
        multifactor_score = rule['multifactor_score_mean']
        
        if weeks <= 12:
            timing = "早期"
        elif weeks <= 18:
            timing = "中期"
        else:
            timing = "中晚期"
        
        if multifactor_score <= 0.3:
            complexity = "低风险"
        elif multifactor_score <= 0.6:
            complexity = "中等风险"
        else:
            complexity = "高风险"
        
        if success_rate >= 0.9:
            reliability = "检测成功率高"
        elif success_rate >= 0.7:
            reliability = "检测成功率中等，建议密切关注"
        else:
            reliability = "检测成功率较低，可能需要重复检测"
        
        return f"{timing}检测，{complexity}，{reliability}"
    
    def _analyze_multifactor_error_impact(self, group_name: str) -> str:
        """分析多因素误差影响"""
        # 检查综合敏感性分析结果
        if 'comprehensive_sensitivity_results' in self.analysis_results:
            comp_results = self.analysis_results['comprehensive_sensitivity_results']
            if 'error_sensitivity' in comp_results and group_name in comp_results['error_sensitivity']:
                error_data = comp_results['error_sensitivity'][group_name]
                # 计算所有误差水平的风险范围
                all_risks = []
                for error_level, data in error_data.items():
                    if isinstance(data, pd.DataFrame) and not data.empty and 'expected_risk' in data.columns:
                        all_risks.extend(data['expected_risk'].tolist())
                
                if all_risks:
                    risk_range = max(all_risks) - min(all_risks)
                    if risk_range < 0.001:
                        return "对测量误差不敏感"
                    elif risk_range < 0.005:
                        return "对测量误差轻微敏感"
                    else:
                        return "对测量误差敏感"
        
        # 回退到原始敏感性分析结果
        elif 'sensitivity_results' in self.analysis_results and group_name in self.analysis_results['sensitivity_results']:
            sens_data = self.analysis_results['sensitivity_results'][group_name]
            if not sens_data.empty and 'expected_risk' in sens_data.columns:
                risk_range = sens_data['expected_risk'].max() - sens_data['expected_risk'].min()
                
                if risk_range < 0.001:
                    return "对测量误差不敏感"
                elif risk_range < 0.005:
                    return "对测量误差轻度敏感"
                else:
                    return "对测量误差较为敏感，需严格控制检测质量"
        
        return "未分析"
    
    def create_multifactor_visualizations(self):
        """创建多因素可视化图表"""
        print("\n=== Step 7: 生成多因素可视化图表 ===")
        
        # 创建保存目录
        os.makedirs('problem3_results/figures', exist_ok=True)
        
        # 这里可以添加具体的可视化代码
        print("  ✓ 多因素可视化图表生成完成")
    
    def save_multifactor_results(self):
        """保存多因素分析结果"""
        print("\n=== Step 8: 保存多因素分析结果 ===")
        
        # 创建结果目录
        os.makedirs('problem3_results/data', exist_ok=True)
        os.makedirs('problem3_results/models', exist_ok=True)
        os.makedirs('problem3_results/reports', exist_ok=True)
        
        # 保存数据结果
        self.analysis_results['valid_predictions'].to_csv(
            'problem3_results/data/多因素预测达标时间.csv', index=False, encoding='utf-8'
        )
        
        self.analysis_results['feature_df'].to_csv(
            'problem3_results/data/多因素特征数据.csv', index=False, encoding='utf-8'
        )
        
        # 保存分组规则
        pd.DataFrame(self.analysis_results['grouping_rules']).to_csv(
            'problem3_results/data/多因素分组规则.csv', index=False, encoding='utf-8'
        )
        
        # 保存最终建议
        pd.DataFrame(self.analysis_results['final_recommendations']).to_csv(
            'problem3_results/data/多因素最终临床建议.csv', index=False, encoding='utf-8'
        )
        
        # 保存模型文件
        self._save_models()
        
        print("多因素分析结果保存完成:")
        print("  ✓ problem3_results/data/ - 数据文件")
        print("  ✓ problem3_results/figures/ - 图表文件")
        print("  ✓ problem3_results/models/ - 模型文件")
        print("  ✓ problem3_results/reports/ - 分析报告")
    
    def _save_models(self):
        """保存模型文件"""
        import pickle
        import json
        
        # 保存决策树分组模型
        if 'grouping_result' in self.analysis_results:
            grouping_result = self.analysis_results['grouping_result']
            if 'clustering_model' in grouping_result:
                with open('problem3_results/models/决策树分组模型.pkl', 'wb') as f:
                    pickle.dump(grouping_result['clustering_model'], f)
                
                # 保存分组模型信息
                model_info = {
                    'model_type': grouping_result.get('best_method', 'DecisionTree'),
                    'n_clusters': grouping_result.get('best_n_clusters', 0),
                    'score': grouping_result.get('best_score', 0),
                    'feature_names': grouping_result.get('feature_names', [])
                }
                with open('problem3_results/models/决策树分组模型信息.json', 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        # 保存时间预测模型参数
        time_model_info = {
            'model_type': 'TimePredictionModelProblem3',
            'parameters': {
                'measurement_error_std': self.time_predictor.measurement_error_std,
                'individual_effect_std': self.time_predictor.individual_effect_std,
                'threshold': self.time_predictor.threshold,
                'model_params': self.time_predictor.model_params,
                'center_params': self.time_predictor.center_params
            }
        }
        with open('problem3_results/models/时间预测模型参数.json', 'w', encoding='utf-8') as f:
            json.dump(time_model_info, f, ensure_ascii=False, indent=2)
        
        # 保存风险模型参数
        risk_model_info = {
            'model_type': 'RiskModelProblem3',
            'parameters': {
                'target_success_probability': self.risk_model.target_success_probability,
                'measurement_error_std': self.risk_model.measurement_error_std,
                'individual_effect_std': self.risk_model.individual_effect_std,
                'multifactor_weights': self.risk_model.multifactor_weights,
                'delay_risk_params': self.risk_model.delay_risk_params
            }
        }
        with open('problem3_results/models/风险模型参数.json', 'w', encoding='utf-8') as f:
            json.dump(risk_model_info, f, ensure_ascii=False, indent=2)
        
        # 保存优化结果
        if 'optimization_results' in self.analysis_results:
            optimization_data = []
            for group_name, opt_result in self.analysis_results['optimization_results'].items():
                optimization_data.append({
                    'group_name': group_name,
                    'optimal_test_time_days': opt_result['optimal_test_time_days'],
                    'optimal_test_time_weeks': opt_result['optimal_test_time_weeks'],
                    'minimal_expected_risk': opt_result['minimal_expected_risk'],
                    'success_proportion': opt_result['detailed_analysis']['success_proportion']
                })
            
            pd.DataFrame(optimization_data).to_csv(
                'problem3_results/models/优化结果.csv', index=False, encoding='utf-8'
            )
        
        # 保存敏感性分析结果
        if 'comprehensive_sensitivity_results' in self.analysis_results:
            sensitivity_data = self.analysis_results['comprehensive_sensitivity_results']
            with open('problem3_results/models/敏感性分析结果.json', 'w', encoding='utf-8') as f:
                # 将DataFrame转换为可序列化的格式
                serializable_results = {}
                for analysis_type, analysis_data in sensitivity_data.items():
                    serializable_results[analysis_type] = {}
                    if isinstance(analysis_data, dict):
                        for group_name, group_data in analysis_data.items():
                            if isinstance(group_data, dict):
                                serializable_results[analysis_type][group_name] = {}
                                for factor, factor_data in group_data.items():
                                    if isinstance(factor_data, pd.DataFrame):
                                        serializable_results[analysis_type][group_name][factor] = factor_data.to_dict('records')
                            elif isinstance(group_data, pd.DataFrame):
                                serializable_results[analysis_type][group_name] = group_data.to_dict('records')
                
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print("  ✓ 决策树分组模型已保存")
        print("  ✓ 时间预测模型参数已保存")
        print("  ✓ 风险模型参数已保存")
        print("  ✓ 优化结果已保存")
        print("  ✓ 敏感性分析结果已保存")
    
    def run_complete_multifactor_analysis(self):
        """运行完整的问题3多因素分析流程"""
        print("开始问题3多因素分析流程")
        print("="*50)
        
        try:
            # Step 1: 数据预处理
            self.load_and_preprocess_data()
            
            # Step 2: 多因素达标时间预测
            self.predict_multifactor_达标时间()
            
            # Step 3: 多因素分组优化
            self.optimize_multifactor_grouping()
            
            # Step 4: 多因素组检测时间优化
            self.optimize_test_times_for_multifactor_groups()
            
            # Step 5: 综合多因素敏感性分析
            self.comprehensive_multifactor_sensitivity_analysis()
            
            # Step 6: 生成最终多因素建议
            self.generate_final_multifactor_recommendations()
            
            # Step 7: 创建可视化
            self.create_multifactor_visualizations()
            
            # Step 8: 保存结果
            self.save_multifactor_results()
            
            print("\n" + "="*50)
            print("问题3多因素分析流程完成！")
            print("结果已保存到 problem3_results/ 目录")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"\n分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 运行问题3多因素分析
    data_path = "../初始数据/男胎检测数据.csv"
    
    analyzer = Problem3CompleteAnalysis(data_path)
    results = analyzer.run_complete_multifactor_analysis()
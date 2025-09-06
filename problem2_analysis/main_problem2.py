"""
问题2主分析模块
整合时间预测、风险建模、决策树分组和优化算法
实现BMI分组和最佳NIPT时点的完整解决方案
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
from time_prediction import TimePredictionModel
from risk_model import RiskModel  
from decision_tree_grouping import DecisionTreeGrouping
from visualization import Problem2Visualizer

# 导入第一问的数据处理器
import sys
import os
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


class Problem2CompleteAnalysis:
    """问题2完整分析流程"""
    
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
        
        # 获取第一问处理好的数据路径
        self.problem1_data_path = os.path.join(
            os.path.dirname(__file__), "..", "problem1_analysis", "results", "data", "processed_data.csv"
        )
        
        # 初始化各个模块（按新思路简化）
        self.time_predictor = TimePredictionModel(self.problem1_data_path)  # 自动拟合简化模型
        self.risk_model = RiskModel()
        self.decision_tree = DecisionTreeGrouping(max_groups=8, min_samples_per_group=15)
        self.visualizer = Problem2Visualizer()
        
        # 存储分析结果
        self.analysis_results = {}
        
        print("问题2分析流程初始化完成")
    
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
        
        print(f"数据预处理完成:")
        print(f"  - 总记录数: {len(self.processed_data)}")
        print(f"  - 唯一孕妇数: {len(self.unique_patients)}")
        print(f"  - BMI范围: {self.unique_patients['孕妇BMI'].min():.1f} - {self.unique_patients['孕妇BMI'].max():.1f}")
        
        return self.processed_data, self.unique_patients
    
    def predict_达标时间_for_all_patients(self):
        """为所有患者预测达标时间和95%概率约束时间"""
        print("\n=== Step 2: 预测达标时间和95%约束时间 ===")
        
        # 批量预测传统达标时间
        prediction_results = self.time_predictor.batch_predict_达标时间(self.unique_patients)
        
        # 为每个患者计算满足95%概率约束的时间（简化版本）
        constraint_results = []
        for _, patient in self.unique_patients.iterrows():
            constraint_time = self.time_predictor.find_time_for_success_probability(
                bmi=patient['孕妇BMI'],  # 只需要BMI参数
                target_prob=self.risk_model.target_success_probability
            )
            
            constraint_results.append({
                '孕妇代码': patient['孕妇代码'],
                '满足95%约束时间': constraint_time,
                '满足95%约束周数': constraint_time / 7 if constraint_time else None
            })
        
        constraint_df = pd.DataFrame(constraint_results)
        
        # 合并结果
        combined_results = prediction_results.merge(constraint_df, on='孕妇代码', how='left')
        
        # 过滤掉无法在合理时间内满足约束的记录
        valid_predictions = combined_results.dropna(subset=['满足95%约束时间'])
        
        print(f"时间预测完成:")
        print(f"  - 总患者数: {len(prediction_results)}")
        print(f"  - 可满足95%约束的患者: {len(valid_predictions)}/{len(prediction_results)}")
        
        if len(valid_predictions) > 0:
            print(f"  - 平均95%约束时间: {valid_predictions['满足95%约束周数'].mean():.1f}周")
            print(f"  - 约束时间范围: {valid_predictions['满足95%约束周数'].min():.1f} - {valid_predictions['满足95%约束周数'].max():.1f}周")
        
        self.analysis_results['prediction_results'] = prediction_results
        self.analysis_results['constraint_results'] = constraint_df
        self.analysis_results['valid_predictions'] = valid_predictions
        
        return valid_predictions
    
    def optimize_bmi_grouping_with_decision_tree(self):
        """使用简化决策树优化BMI分组"""
        print("\n=== Step 3: 简化决策树分组优化 ===")
        
        # 准备简化的决策树特征数据
        feature_df = self.decision_tree.prepare_simplified_features_for_grouping(
            self.analysis_results['valid_predictions'], 
            self.time_predictor, 
            self.risk_model
        )
        
        print(f"特征数据准备完成，共 {len(feature_df)} 个样本")
        
        # 运行简化决策树分组优化
        grouping_result = self.decision_tree.optimize_grouping_with_simplified_decision_tree(feature_df)
        
        print(f"简化决策树分组完成:")
        if grouping_result['best_depth'] is not None:
            print(f"  - 最优树深度: {grouping_result['best_depth']}")
            print(f"  - 交叉验证分数: {grouping_result['best_cv_score']:.4f}")
        print(f"  - 生成组数: {len(grouping_result['groups'])}")
        
        # 简化分组规则生成（直接从分组结果生成）
        grouping_rules = []
        for i, (group_name, group_info) in enumerate(grouping_result['groups'].items()):
            group_data = group_info['data']
            bmi_range = grouping_result['group_stats'][group_name]['bmi_range']
            
            # 计算期望风险（从group_stats中获取）
            expected_risk = grouping_result['group_stats'][group_name].get('expected_minimal_risk', 0.1)
            recommended_time = grouping_result['group_stats'][group_name]['optimal_test_time_mean']
            
            rule = {
                'group_id': i + 1,
                'group_name': f"BMI组{i+1}",
                'bmi_lower_bound': bmi_range[0],
                'bmi_upper_bound': bmi_range[1],
                'bmi_interval_description': f"[{bmi_range[0]:.1f}, {bmi_range[1]:.1f}]",
                'sample_size': len(group_data),
                'recommended_test_time_weeks': recommended_time,
                'earliest_达标时间_weeks': grouping_result['group_stats'][group_name]['earliest_达标时间_mean'],
                'constraint_satisfaction_rate': grouping_result['group_stats'][group_name]['constraint_satisfaction_rate'],
                'expected_risk': expected_risk,
                'clinical_recommendation': self._get_clinical_recommendation(recommended_time)
            }
            grouping_rules.append(rule)
        
        print(f"\n简化BMI分组规则:")
        for rule in grouping_rules:
            print(f"  {rule['group_name']}: BMI {rule['bmi_interval_description']}, "
                  f"平均达标 {rule['earliest_达标时间_weeks']:.1f}周, "
                  f"推荐检测 {rule['recommended_test_time_weeks']:.1f}周, "
                  f"样本数 {rule['sample_size']}")
        
        self.analysis_results['feature_df'] = feature_df
        self.analysis_results['grouping_result'] = grouping_result
        self.analysis_results['grouping_rules'] = grouping_rules
        
        return grouping_rules
    
    def _get_clinical_recommendation(self, test_time_weeks: float) -> str:
        """生成临床建议文本"""
        if test_time_weeks <= 12:
            return f"建议在{test_time_weeks:.1f}周检测，属于早期检测，风险较低"
        elif test_time_weeks <= 18:
            return f"建议在{test_time_weeks:.1f}周检测，属于中期检测，需要密切关注"
        else:
            return f"建议在{test_time_weeks:.1f}周检测，属于中晚期检测，建议提前准备"
    
    def optimize_test_times_for_groups(self):
        """为各BMI组优化检测时间"""
        print("\n=== Step 4: 各组最佳检测时间优化 ===")
        
        optimization_results = {}
        
        for group_name, group_info in self.analysis_results['grouping_result']['groups'].items():
            group_stats = self.analysis_results['grouping_result']['group_stats'][group_name]
            bmi_range = group_stats['bmi_range']
            
            print(f"\n优化 {group_name} (BMI: [{bmi_range[0]:.1f}, {bmi_range[1]:.1f}]):")
            
            # 获取该组的数据
            group_data = group_info['data']
            
            # 转换为原始数据格式用于风险计算
            group_patient_data = []
            for _, row in group_data.iterrows():
                patient_code = row['孕妇代码']
                original_data = self.unique_patients[self.unique_patients['孕妇代码'] == patient_code].iloc[0]
                group_patient_data.append(original_data)
            
            group_patient_df = pd.DataFrame(group_patient_data)
            
            # 优化该组的检测时间（使用原来的方法）
            opt_result = self.risk_model.optimize_test_time_for_group(
                group_patient_df, self.time_predictor, min_week=10, max_week=22
            )
            
            print(f"  - 最优检测时间: {opt_result['optimal_test_time_weeks']:.1f}周")
            print(f"  - 最小期望风险: {opt_result['minimal_expected_risk']:.4f}")
            print(f"  - 达标成功率: {opt_result['detailed_analysis']['success_rate']:.3f}")
            
            optimization_results[group_name] = opt_result
        
        self.analysis_results['optimization_results'] = optimization_results
        
        return optimization_results
    
    def analyze_measurement_error_sensitivity(self):
        """分析检测误差对结果的敏感性"""
        print("\n=== Step 5: 检测误差敏感性分析 ===")
        
        error_range = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
        sensitivity_results = {}
        
        for group_name, group_info in self.analysis_results['grouping_result']['groups'].items():
            print(f"\n分析 {group_name} 的误差敏感性:")
            
            # 获取该组的优化检测时间
            optimal_time_days = self.analysis_results['optimization_results'][group_name]['optimal_test_time_days']
            
            # 获取组数据
            group_data = group_info['data']
            group_patient_data = []
            for _, row in group_data.iterrows():
                patient_code = row['孕妇代码']
                original_data = self.unique_patients[self.unique_patients['孕妇代码'] == patient_code].iloc[0]
                group_patient_data.append(original_data)
            
            group_patient_df = pd.DataFrame(group_patient_data)
            
            # 进行敏感性分析
            sens_result = self.risk_model.analyze_sensitivity_to_error(
                optimal_time_days, group_patient_df, self.time_predictor, error_range
            )
            
            sensitivity_results[group_name] = sens_result
            
            print(f"  - 误差范围: {min(error_range):.3f} - {max(error_range):.3f}")
            print(f"  - 风险变化: {sens_result['expected_risk'].min():.4f} - {sens_result['expected_risk'].max():.4f}")
        
        self.analysis_results['sensitivity_results'] = sensitivity_results
        
        return sensitivity_results
    
    def generate_final_recommendations(self):
        """生成最终的临床建议（包含三个约束条件验证）"""
        print("\n=== Step 6: 生成最终临床建议 ===")
        
        recommendations = []
        
        # 收集所有BMI分组和检测时间用于约束检查
        bmi_groups = []
        detection_times = []
        
        # 获取实际的组名（从grouping_result中）
        actual_group_names = list(self.analysis_results['grouping_result']['groups'].keys())
        
        for i, rule in enumerate(self.analysis_results['grouping_rules']):
            group_name = actual_group_names[i]  # 使用实际的组名
            opt_result = self.analysis_results['optimization_results'][group_name]
            
            # 收集约束检查数据
            # 从bmi_interval_description中提取数值范围
            bmi_interval = rule['bmi_interval_description']
            # 解析类似 "[20.0, 25.0]" 的字符串
            import re
            match = re.search(r'\[([\d.]+),\s*([\d.]+)\]', bmi_interval)
            if match:
                bmi_min = float(match.group(1))
                bmi_max = float(match.group(2))
                bmi_groups.append((bmi_min, bmi_max))
            else:
                # 备用方案：使用默认范围
                bmi_groups.append((20.0, 50.0))
            
            detection_times.append(opt_result['optimal_test_time_weeks'])
            
            recommendation = {
                'group_id': rule['group_id'],
                'group_name': rule['group_name'], 
                'bmi_range': rule['bmi_interval_description'],
                'sample_size': rule['sample_size'],
                'recommended_test_time_weeks': opt_result['optimal_test_time_weeks'],
                'recommended_test_time_days': opt_result['optimal_test_time_days'],
                'expected_minimal_risk': opt_result['minimal_expected_risk'],
                'success_rate': opt_result['detailed_analysis']['success_rate'],
                'constraint_satisfaction_rate': opt_result.get('constraint_satisfaction_rate', 0),
                'within_group_robust_constraint_valid': opt_result.get('within_group_robust_constraint_valid', False),
                'within_group_min_probability': opt_result.get('within_group_min_probability', 0),
                'clinical_advice': self._generate_clinical_advice(opt_result),
                'error_sensitivity': self._analyze_error_impact(group_name)
            }
            
            recommendations.append(recommendation)
        
        # 验证三个约束条件
        print("\n约束条件验证:")
        print("-" * 50)
        
        # 约束1：BMI分段约束
        bmi_constraint_valid = self.risk_model.check_bmi_segmentation_constraints(bmi_groups)
        print(f"约束1 - BMI分段约束: {'✅ 满足' if bmi_constraint_valid else '❌ 不满足'}")
        
        # 约束2：检测时窗约束
        time_constraint_valid = self.risk_model.check_detection_time_constraints(detection_times)
        print(f"约束2 - 检测时窗约束(10-25周): {'✅ 满足' if time_constraint_valid else '❌ 不满足'}")
        
        # 约束3：组内稳妥达标约束
        robust_constraint_count = sum(1 for rec in recommendations if rec['within_group_robust_constraint_valid'])
        print(f"约束3 - 组内稳妥达标约束: {robust_constraint_count}/{len(recommendations)} 组满足")
        
        self.analysis_results['final_recommendations'] = recommendations
        self.analysis_results['constraint_validation'] = {
            'bmi_segmentation_valid': bmi_constraint_valid,
            'detection_time_valid': time_constraint_valid,
            'robust_constraint_satisfied_groups': robust_constraint_count,
            'total_groups': len(recommendations)
        }
        
        # 打印最终建议
        print("\n最终BMI分组和NIPT时点建议:")
        print("="*80)
        for rec in recommendations:
            print(f"{rec['group_name']}: BMI {rec['bmi_range']}")
            print(f"  - 建议检测时间: {rec['recommended_test_time_weeks']:.1f}周 ({rec['recommended_test_time_days']:.0f}天)")
            print(f"  - 最小总风险: {rec['expected_minimal_risk']:.4f}")
            print(f"  - 平均成功概率: {rec['success_rate']:.3f}")
            print(f"  - 95%约束满足率: {rec.get('constraint_satisfaction_rate', 0):.3f}")
            print(f"  - 组内稳妥达标: {'✅' if rec['within_group_robust_constraint_valid'] else '❌'}")
            print(f"  - 组内最小概率: {rec.get('within_group_min_probability', 0):.3f}")
            print(f"  - 临床建议: {rec['clinical_advice']}")
            print(f"  - 误差敏感性: {rec['error_sensitivity']}")
            print()
        
        return recommendations
    
    def _generate_clinical_advice(self, opt_result: Dict) -> str:
        """生成临床建议文本"""
        weeks = opt_result['optimal_test_time_weeks']
        success_rate = opt_result['detailed_analysis']['success_rate']
        
        if weeks <= 12:
            timing = "早期"
        elif weeks <= 18:
            timing = "中期"
        else:
            timing = "中晚期"
        
        if success_rate >= 0.9:
            reliability = "检测成功率高"
        elif success_rate >= 0.7:
            reliability = "检测成功率中等，建议密切关注"
        else:
            reliability = "检测成功率较低，可能需要重复检测"
        
        return f"{timing}检测，{reliability}"
    
    def _analyze_error_impact(self, group_name: str) -> str:
        """分析误差影响"""
        if group_name in self.analysis_results['sensitivity_results']:
            sens_data = self.analysis_results['sensitivity_results'][group_name]
            risk_range = sens_data['expected_risk'].max() - sens_data['expected_risk'].min()
            
            if risk_range < 0.001:
                return "对测量误差不敏感"
            elif risk_range < 0.005:
                return "对测量误差轻度敏感"
            else:
                return "对测量误差较为敏感，需严格控制检测质量"
        return "未分析"
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n=== Step 7: 生成可视化图表 ===")
        
        # 创建保存目录
        os.makedirs('problem2_results/figures', exist_ok=True)
        
        # 1. 达标时间 vs BMI
        fig1 = self.visualizer.plot_达标时间_vs_bmi(
            self.analysis_results['valid_predictions'],
            save_path='problem2_results/figures/达标时间_vs_BMI.png'
        )
        print("  ✓ 生成达标时间vs BMI关系图")
        
        # 2. 决策树可视化
        fig2 = self.decision_tree.visualize_decision_tree(
            save_path='problem2_results/figures/BMI分组决策树.png'
        )
        print("  ✓ 生成BMI分组决策树图")
        
        # 3. 分组结果可视化
        fig3 = self.visualizer.plot_grouping_results(
            self.analysis_results['grouping_rules'],
            None,  # refined_groups参数在方法中未使用，传递None
            save_path='problem2_results/figures/BMI分组结果.png'
        )
        print("  ✓ 生成BMI分组结果图")
        
        # 4. 敏感性分析图
        if self.analysis_results['sensitivity_results']:
            # 取第一组作为示例
            first_group = list(self.analysis_results['sensitivity_results'].keys())[0]
            sens_data = self.analysis_results['sensitivity_results'][first_group]
            
            fig4 = self.visualizer.plot_sensitivity_analysis(
                sens_data,
                save_path='problem2_results/figures/检测误差敏感性分析.png'
            )
            print("  ✓ 生成检测误差敏感性分析图")
        
        # 5. 综合仪表板
        fig5 = self.visualizer.create_comprehensive_dashboard(
            self.analysis_results,
            save_path='problem2_results/figures/问题2综合分析仪表板.png'
        )
        print("  ✓ 生成综合分析仪表板")
        
        print("可视化图表生成完成")
    
    def save_results(self):
        """保存分析结果"""
        print("\n=== Step 8: 保存分析结果 ===")
        
        # 创建结果目录
        os.makedirs('problem2_results/data', exist_ok=True)
        os.makedirs('problem2_results/models', exist_ok=True)
        os.makedirs('problem2_results/reports', exist_ok=True)
        
        # 保存数据结果
        self.analysis_results['valid_predictions'].to_csv(
            'problem2_results/data/预测达标时间.csv', index=False, encoding='utf-8'
        )
        
        self.analysis_results['feature_df'].to_csv(
            'problem2_results/data/决策树特征数据.csv', index=False, encoding='utf-8'
        )
        
        # 保存分组规则
        pd.DataFrame(self.analysis_results['grouping_rules']).to_csv(
            'problem2_results/data/BMI分组规则.csv', index=False, encoding='utf-8'
        )
        
        # 保存最终建议
        pd.DataFrame(self.analysis_results['final_recommendations']).to_csv(
            'problem2_results/data/最终临床建议.csv', index=False, encoding='utf-8'
        )
        
        # 保存模型文件
        self._save_models()
        
        # 保存完整结果（JSON格式）
        # 处理不能序列化的对象
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if key not in ['tree_model', 'grouping_result']:  # 跳过复杂对象
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict('records')
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    try:
                        json.dumps(value)  # 测试是否可序列化
                        serializable_results[key] = value
                    except:
                        serializable_results[key] = str(value)
        
        with open('problem2_results/data/完整分析结果.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 生成分析报告
        self._generate_analysis_report()
        
        print("分析结果保存完成:")
        print("  ✓ problem2_results/data/ - 数据文件")
        print("  ✓ problem2_results/figures/ - 图表文件")
        print("  ✓ problem2_results/models/ - 模型文件")
        print("  ✓ problem2_results/reports/ - 分析报告")
    
    def _save_models(self):
        """保存训练好的模型"""
        import pickle
        import joblib
        
        print("  ✓ 保存决策树模型...")
        
        # 保存决策树模型
        if 'grouping_result' in self.analysis_results and 'tree_model' in self.analysis_results['grouping_result']:
            tree_model = self.analysis_results['grouping_result']['tree_model']
            joblib.dump(tree_model, 'problem2_results/models/decision_tree_model.pkl')
            
            # 保存决策树特征重要性
            feature_importance = self.analysis_results['grouping_result'].get('feature_importance', {})
            with open('problem2_results/models/decision_tree_feature_importance.json', 'w', encoding='utf-8') as f:
                json.dump(feature_importance, f, ensure_ascii=False, indent=2)
        
        print("  ✓ 保存时间预测模型参数...")
        
        # 保存时间预测模型参数
        time_predictor_params = {
            'model_params': self.time_predictor.model_params,
            'threshold': self.time_predictor.threshold,
            'bmi_correlation': getattr(self.time_predictor, 'bmi_correlation', None)
        }
        with open('problem2_results/models/time_prediction_model.json', 'w', encoding='utf-8') as f:
            json.dump(time_predictor_params, f, ensure_ascii=False, indent=2)
        
        print("  ✓ 保存风险模型参数...")
        
        # 保存风险模型参数
        risk_model_params = {
            'target_success_probability': self.risk_model.target_success_probability,
            'measurement_error_std': self.risk_model.measurement_error_std,
            'individual_effect_std': self.risk_model.individual_effect_std,
            'bmi_min': self.risk_model.bmi_min,
            'bmi_max': self.risk_model.bmi_max,
            'detection_min_week': self.risk_model.detection_min_week,
            'detection_max_week': self.risk_model.detection_max_week,
            'delay_risk_params': self.risk_model.delay_risk_params,
            'detection_failure_penalty': self.risk_model.detection_failure_penalty
        }
        with open('problem2_results/models/risk_model.json', 'w', encoding='utf-8') as f:
            json.dump(risk_model_params, f, ensure_ascii=False, indent=2)
        
        print("  ✓ 保存优化结果...")
        
        # 保存优化结果
        if 'optimization_results' in self.analysis_results:
            optimization_summary = {}
            for group_name, opt_result in self.analysis_results['optimization_results'].items():
                optimization_summary[group_name] = {
                    'optimal_test_time_weeks': float(opt_result['optimal_test_time_weeks']),
                    'minimal_expected_risk': float(opt_result['minimal_expected_risk']),
                    'group_success_rate_mean': float(opt_result.get('group_success_rate_mean', 0)),
                    'constraint_satisfaction_rate': float(opt_result.get('constraint_satisfaction_rate', 0)),
                    'within_group_robust_constraint_valid': bool(opt_result.get('within_group_robust_constraint_valid', False)),
                    'sample_size': int(opt_result.get('sample_size', 0))
                }
            
            with open('problem2_results/models/optimization_results.json', 'w', encoding='utf-8') as f:
                json.dump(optimization_summary, f, ensure_ascii=False, indent=2)
        
        print("  ✓ 保存敏感性分析结果...")
        
        # 保存敏感性分析结果
        if 'sensitivity_results' in self.analysis_results:
            sensitivity_summary = {}
            for group_name, sens_data in self.analysis_results['sensitivity_results'].items():
                if isinstance(sens_data, pd.DataFrame):
                    # 转换DataFrame为可序列化的格式
                    records = sens_data.to_dict('records')
                    # 确保所有numpy类型都转换为Python原生类型
                    for record in records:
                        for key, value in record.items():
                            if hasattr(value, 'item'):  # numpy scalar
                                record[key] = value.item()
                            elif isinstance(value, (np.bool_, bool)):
                                record[key] = bool(value)
                            elif isinstance(value, (np.integer, int)):
                                record[key] = int(value)
                            elif isinstance(value, (np.floating, float)):
                                record[key] = float(value)
                    sensitivity_summary[group_name] = records
                else:
                    sensitivity_summary[group_name] = sens_data
            
            with open('problem2_results/models/sensitivity_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(sensitivity_summary, f, ensure_ascii=False, indent=2)
    
    def _generate_analysis_report(self):
        """生成分析报告"""
        report_content = f"""# NIPT最佳时点选择分析报告 - 问题2

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 分析概述

本报告基于决策树算法和风险优化模型，对男胎孕妇进行BMI分组，并为每组确定最佳NIPT检测时点，以最小化孕妇的潜在风险。

### 数据概况
- 总样本数: {len(self.unique_patients)}
- 有效预测样本数: {len(self.analysis_results['valid_predictions'])}
- BMI范围: {self.unique_patients['孕妇BMI'].min():.1f} - {self.unique_patients['孕妇BMI'].max():.1f}
- 平均预测达标时间: {self.analysis_results['valid_predictions']['预测达标周数'].mean():.1f}周

## 2. BMI分组结果

基于决策树算法，将孕妇分为 {len(self.analysis_results['grouping_rules'])} 组：

"""
        
        for rule in self.analysis_results['grouping_rules']:
            report_content += f"""
### {rule['group_name']}
- **BMI范围**: {rule['bmi_interval_description']}  
- **样本数**: {rule['sample_size']}
- **推荐检测时间**: {rule['recommended_test_time_weeks']:.1f}周
- **期望风险**: {rule['expected_risk']:.4f}
"""

        report_content += f"""

## 3. 优化结果总结

"""
        
        for rec in self.analysis_results['final_recommendations']:
            report_content += f"""
### {rec['group_name']}
- **BMI区间**: {rec['bmi_range']}
- **最佳检测时间**: {rec['recommended_test_time_weeks']:.1f}周 ({rec['recommended_test_time_days']:.0f}天)
- **最小期望风险**: {rec['expected_minimal_risk']:.4f}
- **检测成功率**: {rec['success_rate']:.3f}
- **临床建议**: {rec['clinical_advice']}
- **误差敏感性**: {rec['error_sensitivity']}
"""

        report_content += """

## 4. 主要发现

1. **BMI显著影响最佳检测时间**: 高BMI孕妇需要更长时间才能达到Y染色体浓度标准。

2. **个体化检测策略**: 不同BMI组的最优检测时间存在显著差异，支持个体化检测策略。

3. **风险最小化**: 通过优化检测时间，可以显著降低孕妇的潜在风险。

4. **检测误差影响**: 测量误差对结果有一定影响，需要严格的质量控制。

## 5. 临床应用建议

1. **分组检测**: 根据BMI进行分组，采用不同的检测时点策略。

2. **质量控制**: 加强检测过程的质量控制，减少测量误差。

3. **风险评估**: 结合个体BMI和其他因素进行综合风险评估。

4. **动态调整**: 根据实际检测结果，动态调整后续检测计划。

"""

        # 保存报告
        with open('problem2_results/reports/问题2分析报告.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def run_complete_analysis(self):
        """运行完整的问题2分析流程"""
        print("开始问题2完整分析流程")
        print("="*50)
        
        try:
            # Step 1: 数据预处理
            self.load_and_preprocess_data()
            
            # Step 2: 预测达标时间
            self.predict_达标时间_for_all_patients()
            
            # Step 3: 决策树分组
            self.optimize_bmi_grouping_with_decision_tree()
            
            # Step 4: 优化检测时间
            self.optimize_test_times_for_groups()
            
            # Step 5: 敏感性分析
            self.analyze_measurement_error_sensitivity()
            
            # Step 6: 生成最终建议
            self.generate_final_recommendations()
            
            # Step 7: 创建可视化
            self.create_visualizations()
            
            # Step 8: 保存结果
            self.save_results()
            
            print("\n" + "="*50)
            print("问题2分析流程完成！")
            print("结果已保存到 problem2_results/ 目录")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"\n分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 运行问题2完整分析
    data_path = "../初始数据/男胎检测数据.csv"
    
    analyzer = Problem2CompleteAnalysis(data_path)
    results = analyzer.run_complete_analysis()
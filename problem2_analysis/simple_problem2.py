"""
问题2简化解决方案
只基于BMI进行简单分组，不使用多特征决策树
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块
from time_prediction import TimePredictionModel
from risk_model import RiskModel  
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


class SimpleProblem2Analysis:
    """问题2简化分析流程 - 只基于BMI分组"""
    
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
        self.time_predictor = TimePredictionModel()
        self.risk_model = RiskModel()
        self.visualizer = Problem2Visualizer()
        
        # 存储分析结果
        self.analysis_results = {}
        
        print("问题2简化分析流程初始化完成")
    
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
    
    def create_simple_bmi_groups(self, bmi_ranges: List[Tuple[float, float]] = None):
        """
        创建基于BMI的简单分组
        
        Parameters:
        - bmi_ranges: BMI区间列表，默认为标准分组
        """
        if bmi_ranges is None:
            # 标准BMI分组：[20,28), [28,32), [32,36), [36,40), [40,50]
            bmi_ranges = [(20, 28), (28, 32), (32, 36), (36, 40), (40, 50)]
        
        print(f"\n=== Step 2: 创建BMI简单分组 ===")
        print(f"使用BMI区间: {bmi_ranges}")
        
        # 为每个患者分配BMI组
        bmi_groups = {}
        group_stats = {}
        
        for i, (bmi_min, bmi_max) in enumerate(bmi_ranges):
            group_name = f"BMI组{i+1}"
            
            # 筛选该BMI组的患者
            if i == len(bmi_ranges) - 1:  # 最后一组包含最大值
                mask = (self.unique_patients['孕妇BMI'] >= bmi_min) & (self.unique_patients['孕妇BMI'] <= bmi_max)
            else:
                mask = (self.unique_patients['孕妇BMI'] >= bmi_min) & (self.unique_patients['孕妇BMI'] < bmi_max)
            
            group_patients = self.unique_patients[mask].copy()
            
            if len(group_patients) > 0:
                bmi_groups[group_name] = {
                    'bmi_range': (bmi_min, bmi_max),
                    'bmi_interval': f"[{bmi_min}, {bmi_max})" if i < len(bmi_ranges) - 1 else f"[{bmi_min}, {bmi_max}]",
                    'patients': group_patients,
                    'sample_size': len(group_patients),
                    'bmi_mean': group_patients['孕妇BMI'].mean(),
                    'bmi_std': group_patients['孕妇BMI'].std()
                }
                
                group_stats[group_name] = {
                    'bmi_range': (bmi_min, bmi_max),
                    'sample_size': len(group_patients),
                    'bmi_mean': group_patients['孕妇BMI'].mean(),
                    'bmi_std': group_patients['孕妇BMI'].std()
                }
                
                print(f"  {group_name}: BMI {bmi_groups[group_name]['bmi_interval']}, 样本数: {len(group_patients)}")
            else:
                print(f"  {group_name}: BMI {bmi_min}-{bmi_max}, 样本数: 0 (跳过)")
        
        self.analysis_results['bmi_groups'] = bmi_groups
        self.analysis_results['group_stats'] = group_stats
        
        return bmi_groups
    
    def predict_达标时间_for_groups(self):
        """为各BMI组预测达标时间"""
        print(f"\n=== Step 3: 预测各BMI组达标时间 ===")
        
        prediction_results = {}
        
        for group_name, group_info in self.analysis_results['bmi_groups'].items():
            print(f"\n处理 {group_name} (BMI: {group_info['bmi_interval']}):")
            
            group_patients = group_info['patients']
            
            # 为组内每个患者预测达标时间
            individual_predictions = []
            for _, patient in group_patients.iterrows():
                # 预测传统达标时间
                predicted_time = self.time_predictor.solve_达标时间(
                    patient['孕妇BMI'], patient['年龄'], patient['身高'], patient['体重']
                )
                
                # 预测满足95%约束的时间
                constraint_time = self.time_predictor.find_time_for_success_probability(
                    patient['孕妇BMI'], patient['年龄'], patient['身高'], patient['体重'],
                    target_prob=self.risk_model.target_success_probability
                )
                
                individual_predictions.append({
                    '孕妇代码': patient['孕妇代码'],
                    '孕妇BMI': patient['孕妇BMI'],
                    '年龄': patient['年龄'],
                    '身高': patient['身高'],
                    '体重': patient['体重'],
                    '预测达标时间': predicted_time,
                    '预测达标周数': predicted_time / 7 if predicted_time else None,
                    '满足95%约束时间': constraint_time,
                    '满足95%约束周数': constraint_time / 7 if constraint_time else None
                })
            
            # 过滤掉无法满足约束的患者
            valid_predictions = [p for p in individual_predictions if p['满足95%约束时间'] is not None]
            
            if len(valid_predictions) > 0:
                # 计算组内统计
                group_predictions_df = pd.DataFrame(valid_predictions)
                
                prediction_results[group_name] = {
                    'individual_predictions': group_predictions_df,
                    'group_mean_bmi': group_predictions_df['孕妇BMI'].mean(),
                    'group_mean_constraint_time': group_predictions_df['满足95%约束周数'].mean(),
                    'group_std_constraint_time': group_predictions_df['满足95%约束周数'].std(),
                    'valid_sample_size': len(valid_predictions),
                    'total_sample_size': len(individual_predictions)
                }
                
                print(f"  - 有效样本: {len(valid_predictions)}/{len(individual_predictions)}")
                print(f"  - 平均95%约束时间: {group_predictions_df['满足95%约束周数'].mean():.1f}周")
                print(f"  - 约束时间范围: {group_predictions_df['满足95%约束周数'].min():.1f} - {group_predictions_df['满足95%约束周数'].max():.1f}周")
            else:
                print(f"  - 警告: 该组没有患者能满足95%约束条件")
                prediction_results[group_name] = {
                    'individual_predictions': pd.DataFrame(),
                    'valid_sample_size': 0,
                    'total_sample_size': len(individual_predictions)
                }
        
        self.analysis_results['prediction_results'] = prediction_results
        return prediction_results
    
    def optimize_test_times_for_groups(self):
        """为各BMI组优化检测时间"""
        print(f"\n=== Step 4: 优化各BMI组检测时间 ===")
        
        optimization_results = {}
        
        for group_name, group_info in self.analysis_results['bmi_groups'].items():
            if group_name not in self.analysis_results['prediction_results']:
                continue
                
            pred_result = self.analysis_results['prediction_results'][group_name]
            
            if pred_result['valid_sample_size'] == 0:
                print(f"\n跳过 {group_name}: 无有效样本")
                continue
            
            print(f"\n优化 {group_name} (BMI: {group_info['bmi_interval']}):")
            
            # 获取该组的患者数据
            group_patients = pred_result['individual_predictions']
            
            # 优化该组的检测时间
            opt_result = self.risk_model.optimize_test_time_for_group(
                group_patients, self.time_predictor, min_week=10, max_week=25
            )
            
            print(f"  - 最优检测时间: {opt_result['optimal_test_time_weeks']:.1f}周")
            print(f"  - 最小期望风险: {opt_result['minimal_expected_risk']:.4f}")
            print(f"  - 达标成功率: {opt_result['detailed_analysis']['success_rate']:.3f}")
            
            optimization_results[group_name] = opt_result
        
        self.analysis_results['optimization_results'] = optimization_results
        return optimization_results
    
    def generate_final_recommendations(self):
        """生成最终的临床建议"""
        print(f"\n=== Step 5: 生成最终临床建议 ===")
        
        recommendations = []
        
        for group_name, group_info in self.analysis_results['bmi_groups'].items():
            if group_name not in self.analysis_results['optimization_results']:
                continue
            
            opt_result = self.analysis_results['optimization_results'][group_name]
            
            recommendation = {
                'group_name': group_name,
                'bmi_range': group_info['bmi_interval'],
                'sample_size': group_info['sample_size'],
                'valid_sample_size': self.analysis_results['prediction_results'][group_name]['valid_sample_size'],
                'recommended_test_time_weeks': opt_result['optimal_test_time_weeks'],
                'recommended_test_time_days': opt_result['optimal_test_time_days'],
                'expected_minimal_risk': opt_result['minimal_expected_risk'],
                'success_rate': opt_result['detailed_analysis']['success_rate'],
                'constraint_satisfaction_rate': opt_result.get('constraint_satisfaction_rate', 0),
                'clinical_advice': self._generate_clinical_advice(opt_result)
            }
            
            recommendations.append(recommendation)
        
        # 打印最终建议
        print("\n最终BMI分组和NIPT时点建议:")
        print("="*80)
        for rec in recommendations:
            print(f"{rec['group_name']}: BMI {rec['bmi_range']}")
            print(f"  - 样本数: {rec['valid_sample_size']}/{rec['sample_size']}")
            print(f"  - 建议检测时间: {rec['recommended_test_time_weeks']:.1f}周 ({rec['recommended_test_time_days']:.0f}天)")
            print(f"  - 最小总风险: {rec['expected_minimal_risk']:.4f}")
            print(f"  - 平均成功概率: {rec['success_rate']:.3f}")
            print(f"  - 95%约束满足率: {rec.get('constraint_satisfaction_rate', 0):.3f}")
            print(f"  - 临床建议: {rec['clinical_advice']}")
            print()
        
        self.analysis_results['final_recommendations'] = recommendations
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
    
    def create_visualizations(self):
        """创建可视化图表"""
        print(f"\n=== Step 6: 生成可视化图表 ===")
        
        # 创建保存目录
        os.makedirs('problem2_simple_results/figures', exist_ok=True)
        
        # 1. BMI分组结果可视化
        if 'bmi_groups' in self.analysis_results:
            fig1 = self.visualizer.plot_simple_bmi_groups(
                self.analysis_results['bmi_groups'],
                save_path='problem2_simple_results/figures/简单BMI分组结果.png'
            )
            print("  ✓ 生成简单BMI分组结果图")
        
        # 2. 检测时间vs BMI
        if 'prediction_results' in self.analysis_results:
            # 合并所有组的预测结果
            all_predictions = []
            for group_name, pred_result in self.analysis_results['prediction_results'].items():
                if pred_result['valid_sample_size'] > 0:
                    df = pred_result['individual_predictions'].copy()
                    df['group_name'] = group_name
                    all_predictions.append(df)
            
            if all_predictions:
                combined_predictions = pd.concat(all_predictions, ignore_index=True)
                fig2 = self.visualizer.plot_达标时间_vs_bmi(
                    combined_predictions,
                    save_path='problem2_simple_results/figures/达标时间_vs_BMI_简单版.png'
                )
                print("  ✓ 生成达标时间vs BMI关系图")
        
        print("可视化图表生成完成")
    
    def analyze_measurement_error_sensitivity(self):
        """分析检测误差对结果的敏感性"""
        print(f"\n=== Step 6: 检测误差敏感性分析 ===")
        
        error_range = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
        sensitivity_results = {}
        
        for group_name, group_info in self.analysis_results['bmi_groups'].items():
            if group_name not in self.analysis_results['optimization_results']:
                continue
                
            print(f"\n分析 {group_name} 的误差敏感性:")
            
            # 获取该组的优化检测时间
            optimal_time_days = self.analysis_results['optimization_results'][group_name]['optimal_test_time_days']
            
            # 获取组数据
            group_patients = group_info['patients']
            
            # 进行敏感性分析
            try:
                sens_result = self.risk_model.analyze_sensitivity_to_error(
                    optimal_time_days, group_patients, self.time_predictor, error_range
                )
                
                sensitivity_results[group_name] = sens_result
                
                print(f"  - 误差范围: {min(error_range):.3f} - {max(error_range):.3f}")
                print(f"  - 风险变化: {sens_result['expected_risk'].min():.4f} - {sens_result['expected_risk'].max():.4f}")
                print(f"  - 成功率变化: {sens_result['success_rate'].min():.3f} - {sens_result['success_rate'].max():.3f}")
                
            except Exception as e:
                print(f"  - 警告: {group_name} 敏感性分析失败: {str(e)}")
                continue
        
        self.analysis_results['sensitivity_results'] = sensitivity_results
        
        # 生成敏感性分析总结
        self._generate_sensitivity_summary(sensitivity_results)
        
        return sensitivity_results
    
    def _generate_sensitivity_summary(self, sensitivity_results: Dict):
        """生成敏感性分析总结"""
        print(f"\n=== 敏感性分析总结 ===")
        
        if not sensitivity_results:
            print("没有可用的敏感性分析结果")
            return
        
        print("各组对测量误差的敏感性:")
        print("-" * 60)
        
        for group_name, sens_data in sensitivity_results.items():
            if sens_data.empty:
                continue
                
            risk_change = sens_data['expected_risk'].max() - sens_data['expected_risk'].min()
            success_change = sens_data['success_rate'].max() - sens_data['success_rate'].min()
            
            # 判断敏感性等级
            if risk_change < 0.01:
                sensitivity_level = "低敏感"
            elif risk_change < 0.05:
                sensitivity_level = "中等敏感"
            else:
                sensitivity_level = "高敏感"
            
            print(f"{group_name}:")
            print(f"  - 风险变化范围: {risk_change:.4f} ({sensitivity_level})")
            print(f"  - 成功率变化范围: {success_change:.3f}")
            print(f"  - 建议: {'标准质控' if sensitivity_level == '低敏感' else '严格质控'}")
            print()
    
    def create_sensitivity_visualizations(self):
        """创建敏感性分析可视化图表"""
        print(f"\n=== Step 7: 生成敏感性分析图表 ===")
        
        if 'sensitivity_results' not in self.analysis_results:
            print("没有敏感性分析结果，跳过图表生成")
            return
        
        # 创建保存目录
        os.makedirs('problem2_simple_results/figures', exist_ok=True)
        
        sensitivity_results = self.analysis_results['sensitivity_results']
        
        if not sensitivity_results:
            print("没有可用的敏感性分析数据")
            return
        
        # 合并所有组的敏感性数据
        all_sensitivity_data = []
        for group_name, sens_data in sensitivity_results.items():
            if not sens_data.empty:
                sens_data_copy = sens_data.copy()
                sens_data_copy['group_name'] = group_name
                all_sensitivity_data.append(sens_data_copy)
        
        if not all_sensitivity_data:
            print("没有有效的敏感性分析数据")
            return
        
        combined_sensitivity = pd.concat(all_sensitivity_data, ignore_index=True)
        
        # 创建敏感性分析图表
        fig = self.visualizer.plot_sensitivity_analysis(
            combined_sensitivity,
            save_path='problem2_simple_results/figures/检测误差敏感性分析_简单版.png'
        )
        
        print("  ✓ 生成检测误差敏感性分析图")
    
    def save_results(self):
        """保存分析结果"""
        print(f"\n=== Step 9: 保存分析结果 ===")
        
        # 创建结果目录
        os.makedirs('problem2_simple_results/data', exist_ok=True)
        os.makedirs('problem2_simple_results/reports', exist_ok=True)
        
        # 保存最终建议
        if 'final_recommendations' in self.analysis_results:
            pd.DataFrame(self.analysis_results['final_recommendations']).to_csv(
                'problem2_simple_results/data/简单BMI分组建议.csv', index=False, encoding='utf-8'
            )
        
        # 保存各组的详细预测结果
        if 'prediction_results' in self.analysis_results:
            all_predictions = []
            for group_name, pred_result in self.analysis_results['prediction_results'].items():
                if pred_result['valid_sample_size'] > 0:
                    df = pred_result['individual_predictions'].copy()
                    df['group_name'] = group_name
                    all_predictions.append(df)
            
            if all_predictions:
                combined_predictions = pd.concat(all_predictions, ignore_index=True)
                combined_predictions.to_csv(
                    'problem2_simple_results/data/简单分组预测结果.csv', index=False, encoding='utf-8'
                )
        
        # 保存敏感性分析结果
        if 'sensitivity_results' in self.analysis_results:
            sensitivity_data = []
            for group_name, sens_data in self.analysis_results['sensitivity_results'].items():
                if not sens_data.empty:
                    sens_data_copy = sens_data.copy()
                    sens_data_copy['group_name'] = group_name
                    sensitivity_data.append(sens_data_copy)
            
            if sensitivity_data:
                combined_sensitivity = pd.concat(sensitivity_data, ignore_index=True)
                combined_sensitivity.to_csv(
                    'problem2_simple_results/data/敏感性分析结果.csv', index=False, encoding='utf-8'
                )
                
                # 保存为JSON格式
                sensitivity_json = {}
                for group_name, sens_data in self.analysis_results['sensitivity_results'].items():
                    if not sens_data.empty:
                        sensitivity_json[group_name] = sens_data.to_dict('records')
                
                with open('problem2_simple_results/data/敏感性分析结果.json', 'w', encoding='utf-8') as f:
                    json.dump(sensitivity_json, f, ensure_ascii=False, indent=2)
        
        # 生成分析报告
        self._generate_simple_report()
        
        print("分析结果保存完成:")
        print("  ✓ problem2_simple_results/data/ - 数据文件")
        print("  ✓ problem2_simple_results/figures/ - 图表文件")
        print("  ✓ problem2_simple_results/reports/ - 分析报告")
    
    def _generate_simple_report(self):
        """生成简化分析报告"""
        report_content = f"""# NIPT最佳时点选择分析报告 - 问题2简化版

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 分析概述

本报告基于问题2要求，采用简单的BMI区间分组方法，为男胎孕妇确定最佳NIPT检测时点，以最小化孕妇的潜在风险。

### 数据概况
- 总样本数: {len(self.unique_patients)}
- BMI范围: {self.unique_patients['孕妇BMI'].min():.1f} - {self.unique_patients['孕妇BMI'].max():.1f}

## 2. BMI分组结果

基于标准BMI区间，将孕妇分为 {len(self.analysis_results.get('bmi_groups', {}))} 组：

"""
        
        for group_name, group_info in self.analysis_results.get('bmi_groups', {}).items():
            report_content += f"""
### {group_name}
- **BMI范围**: {group_info['bmi_interval']}  
- **样本数**: {group_info['sample_size']}
- **平均BMI**: {group_info['bmi_mean']:.1f}
"""
        
        if 'final_recommendations' in self.analysis_results:
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
"""
        
        # 添加敏感性分析结果
        if 'sensitivity_results' in self.analysis_results:
            report_content += f"""

## 4. 敏感性分析结果

### 检测误差影响评估

各组对测量误差（0.002-0.010范围）的敏感性分析：

"""
            for group_name, sens_data in self.analysis_results['sensitivity_results'].items():
                if sens_data.empty:
                    continue
                    
                risk_change = sens_data['expected_risk'].max() - sens_data['expected_risk'].min()
                success_change = sens_data['success_rate'].max() - sens_data['success_rate'].min()
                
                if risk_change < 0.01:
                    sensitivity_level = "低敏感"
                elif risk_change < 0.05:
                    sensitivity_level = "中等敏感"
                else:
                    sensitivity_level = "高敏感"
                
                report_content += f"""
#### {group_name}
- **风险变化范围**: {risk_change:.4f} ({sensitivity_level})
- **成功率变化范围**: {success_change:.3f}
- **质量控制建议**: {'标准质控流程即可' if sensitivity_level == '低敏感' else '需要更严格的质量控制'}
"""
        
        report_content += """

## 5. 主要发现

1. **BMI显著影响最佳检测时间**: 高BMI孕妇需要更长时间才能达到Y染色体浓度标准。

2. **简单分组策略**: 基于BMI的简单区间分组能够有效区分不同风险群体。

3. **风险最小化**: 通过优化检测时间，可以显著降低孕妇的潜在风险。

4. **误差敏感性**: 不同BMI组对测量误差的敏感性存在差异，需要针对性的质量控制策略。

## 6. 临床应用建议

1. **分组检测**: 根据BMI进行简单分组，采用不同的检测时点策略。

2. **质量控制**: 加强检测过程的质量控制，减少测量误差。

3. **风险评估**: 结合个体BMI进行风险评估。

4. **动态调整**: 根据实际检测结果，动态调整后续检测计划。

5. **误差控制**: 根据敏感性分析结果，对不同BMI组采用相应的质量控制标准。

"""
        
        # 保存报告
        with open('problem2_simple_results/reports/问题2简化分析报告.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def run_complete_analysis(self):
        """运行完整的问题2简化分析流程"""
        print("开始问题2简化分析流程")
        print("="*50)
        
        try:
            # Step 1: 数据预处理
            self.load_and_preprocess_data()
            
            # Step 2: 创建BMI简单分组
            self.create_simple_bmi_groups()
            
            # Step 3: 预测达标时间
            self.predict_达标时间_for_groups()
            
            # Step 4: 优化检测时间
            self.optimize_test_times_for_groups()
            
            # Step 5: 生成最终建议
            self.generate_final_recommendations()
            
            # Step 6: 检测误差敏感性分析
            self.analyze_measurement_error_sensitivity()
            
            # Step 7: 创建可视化
            self.create_visualizations()
            
            # Step 8: 创建敏感性分析图表
            self.create_sensitivity_visualizations()
            
            # Step 9: 保存结果
            self.save_results()
            
            print("\n" + "="*50)
            print("问题2简化分析流程完成！")
            print("结果已保存到 problem2_simple_results/ 目录")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"\n分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 运行问题2简化分析
    data_path = "../初始数据/男胎检测数据.csv"
    
    analyzer = SimpleProblem2Analysis(data_path)
    results = analyzer.run_complete_analysis()
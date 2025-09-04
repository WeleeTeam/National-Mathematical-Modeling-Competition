"""
问题1主分析程序
整合所有模块，完成NIPT Y染色体浓度的混合效应模型分析
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import NIPTDataProcessor
from mixed_effects_model import NIPTMixedEffectsModel
from visualization import NIPTVisualizer
from results_manager import NIPTResultsManager


class Problem1Analysis:
    """问题1完整分析流程"""
    
    def __init__(self, data_path: str, results_dir: str = "results/"):
        self.data_path = data_path
        self.results_dir = results_dir
        
        # 初始化各模块
        self.processor = NIPTDataProcessor()
        self.model_analyzer = NIPTMixedEffectsModel()
        self.visualizer = NIPTVisualizer(output_dir=os.path.join(results_dir, "figures/"))
        self.results_manager = NIPTResultsManager(results_dir=results_dir)
        
        # 存储分析结果
        self.analysis_results = {}
        self.processed_data = None
        
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        
        print("="*60)
        print("开始NIPT Y染色体浓度混合效应模型分析")
        print("="*60)
        
        # 第一步：数据处理
        print("\n步骤1: 数据加载与预处理")
        self.data_preprocessing()
        
        # 第二步：探索性数据分析
        print("\n步骤2: 探索性数据分析")
        self.exploratory_data_analysis()
        
        # 第三步：相关性分析
        print("\n步骤3: 相关性分析")
        self.correlation_analysis()
        
        # 第四步：混合效应模型拟合
        print("\n步骤4: 混合效应模型拟合")
        self.fit_mixed_effects_models()
        
        # 第五步：模型比较与选择
        print("\n步骤5: 模型比较与选择")
        self.model_comparison()
        
        # 第六步：显著性检验
        print("\n步骤6: 显著性检验")
        self.significance_testing()
        
        # 第七步：模型诊断
        print("\n步骤7: 模型诊断")
        self.model_diagnostics()
        
        # 第八步：结果保存与报告生成
        print("\n步骤8: 结果保存与报告生成")
        self.save_all_results()
        
        print("\n" + "="*60)
        print("分析完成！所有结果已保存到:", self.results_dir)
        print("="*60)
    
    def data_preprocessing(self):
        """数据预处理"""
        
        # 加载数据
        raw_data = self.processor.load_data(self.data_path)
        
        # 清洗数据
        cleaned_data = self.processor.clean_data()
        
        # 准备纵向数据格式
        self.processed_data = self.processor.prepare_longitudinal_format(cleaned_data)
        
        # 获取描述性统计
        summary_stats = self.processor.get_summary_statistics(self.processed_data)
        self.analysis_results['summary_stats'] = summary_stats
        
        # 保存处理后的数据
        self.processor.save_processed_data(
            os.path.join(self.results_dir, "data/processed_data.csv")
        )
        
        print(f"数据预处理完成：{len(self.processed_data)}行，{self.processed_data['孕妇代码'].nunique()}个个体")
    
    def exploratory_data_analysis(self):
        """探索性数据分析"""
        
        # 绘制个体轨迹图
        self.visualizer.plot_individual_trajectories(
            self.processed_data, 
            n_subjects=25,
            save_path=self.results_manager.get_figure_path("individual_trajectories.png")
        )
        
        # 绘制分布分析图
        self.visualizer.plot_distribution_analysis(
            self.processed_data,
            save_path=self.results_manager.get_figure_path("distribution_analysis.png")
        )
        
        # 绘制散点图关系
        self.visualizer.plot_scatter_relationships(
            self.processed_data,
            save_path=self.results_manager.get_figure_path("scatter_relationships.png")
        )
        
        print("探索性数据分析完成，图表已保存")
    
    def correlation_analysis(self):
        """相关性分析"""
        
        # 计算相关性矩阵
        numeric_vars = ['Y染色体浓度', 'gestational_days', '孕妇BMI', '年龄', 
                       '身高', '体重']
        # 使用数值化后的计数列（若存在）
        for optional_col in ['怀孕次数_num', '生产次数_num', '检测抽血次数_num']:
            if optional_col in self.processed_data.columns:
                numeric_vars.append(optional_col)
        
        # 强制转换为数值，无法转换设为NaN
        corr_input = self.processed_data[numeric_vars].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = corr_input.corr()
        self.analysis_results['correlation_matrix'] = correlation_matrix
        
        # 绘制相关性热力图
        self.visualizer.plot_correlation_heatmap(
            self.processed_data,
            save_path=self.results_manager.get_figure_path("correlation_heatmap.png")
        )
        
        print("相关性分析完成")
        print("主要相关系数：")
        def safe_corr(a, b):
            try:
                val = correlation_matrix.loc[a, b]
                return f"{val:.3f}" if pd.notna(val) else "NA"
            except Exception:
                return "NA"

        print(f"Y浓度 vs 孕周数: {safe_corr('Y染色体浓度','gestational_days')}")
        print(f"Y浓度 vs BMI: {safe_corr('Y染色体浓度','孕妇BMI')}")
        print(f"Y浓度 vs 年龄: {safe_corr('Y染色体浓度','年龄')}")
    
    def fit_mixed_effects_models(self):
        """拟合多个混合效应模型"""
        
        print("拟合模型中...")
        
        # 模型1：基础随机截距模型
        result1 = self.model_analyzer.fit_basic_model(
            self.processed_data, 
            model_name="basic_random_intercept"
        )
        
        # 模型2：随机截距+随机斜率模型
        result2 = self.model_analyzer.fit_random_slope_model(
            self.processed_data,
            model_name="random_slope"
        )
        
        # 模型3：包含交互效应的模型
        result3 = self.model_analyzer.fit_interaction_model(
            self.processed_data,
            model_name="interaction_effects"
        )
        
        # 模型4：非线性时间效应模型
        result4 = self.model_analyzer.fit_nonlinear_model(
            self.processed_data,
            model_name="nonlinear_time"
        )
        
        print(f"共拟合了{len(self.model_analyzer.results)}个模型")
    
    def model_comparison(self):
        """模型比较与选择"""
        
        # 比较模型
        comparison_df = self.model_analyzer.compare_models()
        self.analysis_results['model_comparison'] = comparison_df
        
        # 绘制模型比较图
        if not comparison_df.empty:
            self.visualizer.plot_model_comparison(
                comparison_df,
                save_path=self.results_manager.get_figure_path("model_comparison.png")
            )
        
        print("模型比较完成")
    
    def significance_testing(self):
        """显著性检验"""
        
        # 获取最佳模型
        best_model_name = self.model_analyzer.best_model
        
        if best_model_name:
            print(f"分析最佳模型: {best_model_name}")
            
            # 固定效应检验
            fixed_effects_df = self.model_analyzer.test_fixed_effects(best_model_name)
            self.analysis_results['best_model_fixed_effects'] = fixed_effects_df
            
            # 随机效应分析
            random_effects_info = self.model_analyzer.test_random_effects(best_model_name)
            self.analysis_results['best_model_random_effects'] = random_effects_info
            
            # 计算R平方
            r_squared_info = self.model_analyzer.calculate_r_squared(
                best_model_name, 
                self.processed_data
            )
            self.analysis_results['best_model_r_squared'] = r_squared_info
            
            # 绘制固定效应图
            if fixed_effects_df is not None:
                self.visualizer.plot_fixed_effects(
                    fixed_effects_df,
                    save_path=self.results_manager.get_figure_path("fixed_effects.png")
                )
            
            print("显著性检验完成")
            print("\n主要固定效应结果:")
            if fixed_effects_df is not None:
                for _, row in fixed_effects_df.iterrows():
                    if row['Variable'] != 'const':
                        print(f"{row['Variable']}: 系数={row['Coefficient']:.6f}, "
                              f"p值={row['p_value']:.6f}, 显著性={row['Significance']}")
        else:
            print("没有找到最佳模型")
    
    def model_diagnostics(self):
        """模型诊断"""
        
        best_model_name = self.model_analyzer.best_model
        
        if best_model_name:
            # 进行模型诊断
            diagnostics = self.model_analyzer.perform_model_diagnostics(
                best_model_name, 
                self.processed_data
            )
            self.analysis_results['best_model_diagnostics'] = diagnostics
            
            # 绘制诊断图
            if diagnostics:
                self.visualizer.plot_model_diagnostics(
                    diagnostics,
                    best_model_name,
                    save_path=self.results_manager.get_figure_path("model_diagnostics.png")
                )
            
            # 预测个体轨迹
            predictions_df = self.model_analyzer.predict_individual_trajectories(
                best_model_name,
                self.processed_data,
                subject_ids=None  # 使用前10个个体
            )
            
            if predictions_df is not None:
                self.analysis_results['model_predictions'] = predictions_df
                
                # 绘制预测vs观测图
                self.visualizer.plot_predicted_vs_observed(
                    predictions_df,
                    save_path=self.results_manager.get_figure_path("predicted_vs_observed.png")
                )
            
            print("模型诊断完成")
        else:
            print("没有最佳模型进行诊断")
    
    def save_all_results(self):
        """保存所有分析结果"""
        
        # 保存描述性统计
        if 'summary_stats' in self.analysis_results:
            self.results_manager.save_summary_statistics(
                self.analysis_results['summary_stats']
            )
        
        # 保存相关性矩阵
        if 'correlation_matrix' in self.analysis_results:
            self.results_manager.save_correlation_matrix(
                self.analysis_results['correlation_matrix']
            )
        
        # 保存模型结果
        self.results_manager.save_model_results(
            self.model_analyzer.results
        )
        
        # 保存模型比较
        if 'model_comparison' in self.analysis_results:
            self.results_manager.save_model_comparison(
                self.analysis_results['model_comparison']
            )
        
        # 保存固定效应结果
        if 'best_model_fixed_effects' in self.analysis_results:
            fixed_effects_dict = {
                self.model_analyzer.best_model: self.analysis_results['best_model_fixed_effects']
            }
            self.results_manager.save_fixed_effects_results(fixed_effects_dict)
        
        # 保存预测结果
        if 'model_predictions' in self.analysis_results:
            self.results_manager.save_predictions(
                self.analysis_results['model_predictions']
            )
        
        # 生成分析报告
        self.results_manager.generate_analysis_report(self.analysis_results)
        
        # 创建结果摘要
        self.results_manager.create_results_summary()
        
        print("所有结果已保存完成")
    
    def get_analysis_summary(self) -> dict:
        """获取分析摘要"""
        
        summary = {
            'data_info': {
                'total_observations': len(self.processed_data) if self.processed_data is not None else 0,
                'unique_subjects': self.processed_data['孕妇代码'].nunique() if self.processed_data is not None else 0,
                '达标率': self.analysis_results.get('summary_stats', {}).get('y_concentration_stats', {}).get('达标率', 'N/A')
            },
            'best_model': self.model_analyzer.best_model,
            'model_performance': self.analysis_results.get('best_model_r_squared', {}),
            'key_findings': self._extract_key_findings()
        }
        
        return summary
    
    def _extract_key_findings(self) -> list:
        """提取关键发现"""
        
        findings = []
        
        # 相关性发现
        if 'correlation_matrix' in self.analysis_results:
            corr_matrix = self.analysis_results['correlation_matrix']
            y_gest_corr = corr_matrix.loc['Y染色体浓度', 'gestational_days']
            y_bmi_corr = corr_matrix.loc['Y染色体浓度', '孕妇BMI']
            
            findings.append(f"Y染色体浓度与孕周数呈{('强' if abs(y_gest_corr) > 0.7 else '中等' if abs(y_gest_corr) > 0.3 else '弱')}正相关 (r={y_gest_corr:.3f})")
            findings.append(f"Y染色体浓度与BMI呈{('强' if abs(y_bmi_corr) > 0.7 else '中等' if abs(y_bmi_corr) > 0.3 else '弱')}负相关 (r={y_bmi_corr:.3f})")
        
        # 模型发现
        if 'best_model_fixed_effects' in self.analysis_results:
            fe_df = self.analysis_results['best_model_fixed_effects']
            significant_effects = fe_df[fe_df['Significance'] != 'ns']
            
            findings.append(f"发现{len(significant_effects)}个显著的固定效应")
            
            for _, row in significant_effects.iterrows():
                if row['Variable'] != 'const':
                    effect_direction = "正向" if row['Coefficient'] > 0 else "负向"
                    findings.append(f"{row['Variable']}对Y染色体浓度有显著{effect_direction}影响 (p={row['p_value']:.4f})")
        
        return findings


def main():
    """主函数"""
    
    # 设置数据路径
    data_path = "../初始数据/男胎检测数据.csv"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 - {data_path}")
        print("请确保数据文件路径正确")
        return
    
    # 创建分析实例
    analysis = Problem1Analysis(data_path, results_dir="results/")
    
    try:
        # 运行完整分析
        analysis.run_complete_analysis()
        
        # 显示分析摘要
        summary = analysis.get_analysis_summary()
        print("\n" + "="*60)
        print("分析摘要")
        print("="*60)
        
        print(f"数据规模: {summary['data_info']['total_observations']}个观测，{summary['data_info']['unique_subjects']}个个体")
        print(f"总体达标率: {summary['data_info']['达标率']:.2%}")
        print(f"最佳模型: {summary['best_model']}")
        
        if summary['model_performance']:
            print(f"模型表现: R² = {summary['model_performance'].get('Pseudo_R2', 'N/A'):.4f}")
        
        print("\n关键发现:")
        for i, finding in enumerate(summary['key_findings'], 1):
            print(f"{i}. {finding}")
        
        print("\n分析完成！结果文件位于 'results/' 目录")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
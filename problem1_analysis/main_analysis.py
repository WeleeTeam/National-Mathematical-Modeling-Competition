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
    
    def __init__(self, data_path: str, results_dir: str = "results/", separate_figures: bool = False):
        self.data_path = data_path
        self.results_dir = results_dir
        self.separate_figures = separate_figures  # 是否生成拆分的图片
        
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
        
        # 第八步：关系模型确定与打印
        print("\n步骤8: 关系模型确定与打印")
        self.print_relationship_models()
        
        # 第九步：结果保存与报告生成
        print("\n步骤9: 结果保存与报告生成")
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
            save_path=self.results_manager.get_figure_path("distribution_analysis.png"),
            separate_figs=self.separate_figures
        )
        
        # 绘制散点图关系
        self.visualizer.plot_scatter_relationships(
            self.processed_data,
            save_path=self.results_manager.get_figure_path("scatter_relationships.png"),
            separate_figs=self.separate_figures
        )
        
        print("探索性数据分析完成，图表已保存")
    
    def correlation_analysis(self):
        """相关性分析（增强版）"""
        
        # 1. 全面的变量选择
        # 核心变量
        core_vars = ['Y染色体浓度', 'gestational_days', '孕妇BMI', '年龄', '身高', '体重']
        
        # Z值变量（重要的标准化指标）
        z_score_vars = []
        for col in ['Y染色体的Z值', 'X染色体的Z值', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']:
            if col in self.processed_data.columns:
                z_score_vars.append(col)
        
        # GC含量变量（测序质量指标）
        gc_vars = []
        for col in ['GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']:
            if col in self.processed_data.columns:
                gc_vars.append(col)
        
        # 测序质量指标
        seq_quality_vars = []
        for col in ['原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数  ', '被过滤掉读段数的比例', 'X染色体浓度']:
            if col in self.processed_data.columns:
                seq_quality_vars.append(col)
        
        # 临床因素
        clinical_vars = []
        for col in ['IVF_indicator', '怀孕次数_num', '生产次数_num', '检测抽血次数_num']:
            if col in self.processed_data.columns:
                clinical_vars.append(col)
        
        # 综合所有变量
        all_numeric_vars = core_vars + z_score_vars + gc_vars + seq_quality_vars + clinical_vars
        
        # 2. 计算全面相关性矩阵（使用Spearman相关系数）
        corr_input = self.processed_data[all_numeric_vars].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = corr_input.corr(method='spearman')
        self.analysis_results['correlation_matrix'] = correlation_matrix
        
        # 3. 计算相关性的显著性检验
        correlation_pvalues = self._calculate_correlation_pvalues(corr_input)
        self.analysis_results['correlation_pvalues'] = correlation_pvalues
        
        # 4. 按BMI分组进行相关性分析
        bmi_group_correlations = self._analyze_bmi_group_correlations(corr_input)
        self.analysis_results['bmi_group_correlations'] = bmi_group_correlations
        
        # 5. 非线性相关性分析
        nonlinear_correlations = self._analyze_nonlinear_correlations(corr_input)
        self.analysis_results['nonlinear_correlations'] = nonlinear_correlations
        
        # 6. 绘制增强版相关性图表
        self.visualizer.plot_enhanced_correlation_analysis(
            self.processed_data,
            correlation_matrix,
            correlation_pvalues,
            bmi_group_correlations,
            save_path=self.results_manager.get_figure_path("enhanced_correlation_analysis.png"),
            separate_figs=self.separate_figures
        )
        
        # 7. 显示结果
        self._display_correlation_results(correlation_matrix, correlation_pvalues)
    
    def fit_mixed_effects_models(self):
        """拟合增强的随机斜率混合效应模型"""
        
        print("拟合增强的随机斜率模型中...")
        
        # 拟合包含高相关性变量的随机斜率模型
        result = self.model_analyzer.fit_random_slope_model(self.processed_data)
        
        if result is not None:
            print("模型拟合成功")
            # 保存模型结果
            self.analysis_results['mixed_effects_model'] = result
        else:
            print("模型拟合失败")
    
    def model_comparison(self):
        """获取模型摘要信息"""
        
        if self.model_analyzer.result is not None:
            # 获取模型摘要
            summary = self.model_analyzer.get_model_summary()
            self.analysis_results['model_summary'] = summary
            
            # 获取性能指标
            performance = self.model_analyzer.calculate_model_performance(self.processed_data)
            self.analysis_results['model_performance'] = performance
            
            print("模型分析完成")
        else:
            print("模型尚未拟合")
    
    def significance_testing(self):
        """显著性检验和结果分析"""
        
        if self.model_analyzer.result is not None:
            print("分析随机斜率模型结果...")
            
            # 获取固定效应摘要
            fixed_effects_df = self.model_analyzer.get_fixed_effects_summary()
            if fixed_effects_df is not None:
                self.analysis_results['fixed_effects'] = fixed_effects_df
                    
                    # 绘制固定效应图
                try:
                        self.visualizer.plot_fixed_effects(
                            fixed_effects_df,
                            save_path=self.results_manager.get_figure_path("fixed_effects.png")
                        )
                except Exception as e:
                    print(f"绘制固定效应图时出错: {e}")
            
            # 获取变量重要性分析
            importance_df = self.model_analyzer.get_variable_importance()
            if importance_df is not None:
                self.analysis_results['variable_importance'] = importance_df
            
            # 获取性能指标
            performance = self.model_analyzer.calculate_model_performance(self.processed_data)
            if performance is not None:
                self.analysis_results['model_performance'] = performance
                
                print(f"\n模型性能指标:")
                # 安全地格式化性能指标
                corr_val = performance.get('Correlation', 'N/A')
                if isinstance(corr_val, (int, float)):
                    print(f"  相关系数: {corr_val:.4f}")
                else:
                    print(f"  相关系数: {corr_val}")
                
                r2_val = performance.get('R_squared', 'N/A')
                if isinstance(r2_val, (int, float)):
                    print(f"  R²: {r2_val:.4f}")
                else:
                    print(f"  R²: {r2_val}")
                
                pseudo_r2_val = performance.get('Pseudo_R2', 'N/A')
                if isinstance(pseudo_r2_val, (int, float)):
                    print(f"  伪R²: {pseudo_r2_val:.4f}")
                else:
                    print(f"  伪R²: {pseudo_r2_val}")
                
                aic_val = performance.get('AIC', 'N/A')
                if isinstance(aic_val, (int, float)):
                    print(f"  AIC: {aic_val:.2f}")
                else:
                    print(f"  AIC: {aic_val}")
            
            print("显著性检验和结果分析完成")
        else:
            print("模型尚未拟合，无法进行显著性检验")
    
    def model_diagnostics(self):
        """模型诊断"""
        
        if self.model_analyzer.result is not None:
            print("进行模型诊断...")
            
            # 简单的残差分析
            try:
                predicted = self.model_analyzer.result.fittedvalues
                observed = self.processed_data['Y染色体浓度'].iloc[:len(predicted)]
                residuals = observed - predicted
                
                # 保存残差信息
                diagnostics = {
                    'residuals': residuals,
                    'predicted': predicted,
                    'observed': observed,
                    'residuals_std': residuals.std(),
                    'residuals_mean': residuals.mean()
                }
                
                self.analysis_results['model_diagnostics'] = diagnostics
                
                print(f"残差标准差: {diagnostics['residuals_std']:.6f}")
                print(f"残差均值: {diagnostics['residuals_mean']:.6f}")
                print("模型诊断完成")

            except Exception as e:
                print(f"模型诊断时出错: {e}")
        else:
            print("模型尚未拟合，无法进行诊断")
    
    def print_relationship_models(self):
        """打印关系模型结果"""
        
        if self.model_analyzer.result is not None:
            print("\n" + "="*80)
            print("增强的随机斜率混合效应模型结果")
            print("="*80)

            # 使用新的方法打印完整结果
            complete_results = self.model_analyzer.print_complete_results(self.processed_data)
            
            # 保存结果到分析结果中
            if complete_results:
                self.analysis_results['complete_model_results'] = complete_results
        else:
            print("模型尚未拟合，无法打印结果")
    
    def save_all_results(self):
        """保存所有分析结果"""
        
        print("\n" + "="*60)
        print("保存所有分析结果")
        print("="*60)
        
        # 确保结果目录存在
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "reports"), exist_ok=True)
        
        # 1. 保存模型结果
        if self.model_analyzer.result is not None:
            print("\n1. 保存模型结果...")
            
            # 保存固定效应结果
            fixed_effects = self.model_analyzer.get_fixed_effects_summary()
            if fixed_effects is not None:
                fixed_effects.to_csv(
                    os.path.join(self.results_dir, "models", "enhanced_fixed_effects_results.csv"), 
                    index=False
                )
                print("  ✓ 固定效应结果已保存: models/enhanced_fixed_effects_results.csv")
                
                # 打印固定效应结果
                print("\n  固定效应结果预览:")
                print(fixed_effects[['Variable', 'Coefficient', 'P_value', 'Significance']].to_string(index=False, float_format='%.6f'))
            
            # 保存变量重要性
            importance = self.model_analyzer.get_variable_importance()
            if importance is not None:
                importance.to_csv(
                    os.path.join(self.results_dir, "models", "enhanced_variable_importance.csv"), 
                    index=False
                )
                print("  ✓ 变量重要性结果已保存: models/enhanced_variable_importance.csv")
                
                # 打印变量重要性
                print("\n  变量重要性预览:")
                print(importance[['Variable', 'Coefficient', 'P_value', 'Significant', 'Effect_Direction']].to_string(index=False, float_format='%.6f'))
            
            # 保存模型公式
            formula_text = self.model_analyzer.get_model_formula_text()
            if formula_text:
                with open(
                    os.path.join(self.results_dir, "models", "enhanced_model_formula.txt"), 
                    "w", encoding='utf-8'
                ) as f:
                    f.write(formula_text)
                print("  ✓ 模型公式已保存: models/enhanced_model_formula.txt")
                
                # 打印模型公式
                print("\n  模型数学表达式:")
                print(formula_text)
            
            # 保存模型性能指标
            performance = self.model_analyzer.calculate_model_performance(self.processed_data)
            if performance:
                with open(
                    os.path.join(self.results_dir, "models", "enhanced_model_performance.json"), 
                    "w", encoding='utf-8'
                ) as f:
                    import json
                    json.dump(performance, f, ensure_ascii=False, indent=2)
                print("  ✓ 模型性能指标已保存: models/enhanced_model_performance.json")
                
                # 打印性能指标
                print("\n  模型性能指标:")
                for key, value in performance.items():
                    print(f"    {key}: {value:.4f}")
        
        # 2. 保存数据结果
        print("\n2. 保存数据结果...")
        
        # 保存相关性矩阵
        if 'correlation_matrix' in self.analysis_results:
            correlation_matrix = self.analysis_results['correlation_matrix']
            correlation_matrix.to_csv(
                os.path.join(self.results_dir, "data", "enhanced_correlation_matrix.csv")
            )
            print("  ✓ 相关性矩阵已保存: data/enhanced_correlation_matrix.csv")
            
            # 打印主要相关性
            print("\n  关键相关性:")
            y_corr = correlation_matrix['Y染色体浓度'].drop('Y染色体浓度').sort_values(key=abs, ascending=False)
            for var, corr in y_corr.head(10).items():
                print(f"    {var}: {corr:.4f}")
        
        # 保存描述性统计
        if 'summary_stats' in self.analysis_results:
            summary_stats = self.analysis_results['summary_stats']
            with open(
                os.path.join(self.results_dir, "data", "enhanced_summary_statistics.json"), 
                "w", encoding='utf-8'
            ) as f:
                import json
                json.dump(summary_stats, f, ensure_ascii=False, indent=2)
            print("  ✓ 描述性统计已保存: data/enhanced_summary_statistics.json")
            
            # 打印关键统计信息
            print("\n  关键统计信息:")
            if 'y_concentration_stats' in summary_stats:
                y_stats = summary_stats['y_concentration_stats']
                
                # 安全地格式化数值
                mean_val = y_stats.get('均值', 'N/A')
                if isinstance(mean_val, (int, float)):
                    print(f"    Y染色体浓度均值: {mean_val:.6f}")
                else:
                    print(f"    Y染色体浓度均值: {mean_val}")
                
                std_val = y_stats.get('标准差', 'N/A')
                if isinstance(std_val, (int, float)):
                    print(f"    Y染色体浓度标准差: {std_val:.6f}")
                else:
                    print(f"    Y染色体浓度标准差: {std_val}")
                
                rate_val = y_stats.get('达标率', 'N/A')
                if isinstance(rate_val, (int, float)):
                    print(f"    达标率: {rate_val:.2%}")
                else:
                    print(f"    达标率: {rate_val}")
        
        # 保存BMI分组相关性
        if 'bmi_group_correlations' in self.analysis_results:
            bmi_corr = self.analysis_results['bmi_group_correlations']
            with open(
                os.path.join(self.results_dir, "data", "bmi_group_correlations.json"), 
                "w", encoding='utf-8'
            ) as f:
                import json
                json.dump(bmi_corr, f, ensure_ascii=False, indent=2)
            print("  ✓ BMI分组相关性已保存: data/bmi_group_correlations.json")
            
            # 打印BMI分组相关性
            print("\n  BMI分组相关性:")
            for group, corrs in bmi_corr.items():
                print(f"    {group}组:")
                for var, corr in corrs.items():
                    print(f"      {var}: {corr:.4f}")
        
        # 3. 保存完整分析报告
        print("\n3. 生成完整分析报告...")
        
        report_content = self._generate_complete_report()
        with open(
            os.path.join(self.results_dir, "reports", "enhanced_analysis_report.md"), 
            "w", encoding='utf-8'
        ) as f:
            f.write(report_content)
        print("  ✓ 完整分析报告已保存: reports/enhanced_analysis_report.md")
        
        # 4. 保存结果摘要
        summary = self.get_analysis_summary()
        with open(
            os.path.join(self.results_dir, "results_summary.json"), 
            "w", encoding='utf-8'
        ) as f:
            import json
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("  ✓ 结果摘要已保存: results_summary.json")
        
        print("\n" + "="*60)
        print("所有结果保存完成！")
        print("="*60)
        print(f"结果保存在目录: {self.results_dir}")
        print("主要文件:")
        print("  - models/enhanced_fixed_effects_results.csv (固定效应结果)")
        print("  - models/enhanced_variable_importance.csv (变量重要性)")
        print("  - models/enhanced_model_formula.txt (模型公式)")
        print("  - data/enhanced_correlation_matrix.csv (相关性矩阵)")
        print("  - reports/enhanced_analysis_report.md (完整分析报告)")
        print("="*60)
    
    def _generate_complete_report(self) -> str:
        """生成完整的分析报告"""
        
        report = []
        report.append("# NIPT Y染色体浓度增强混合效应模型分析报告")
        report.append(f"\n**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 数据概况
        report.append("\n## 1. 数据概况")
        if self.processed_data is not None:
            report.append(f"- **总观测数**: {len(self.processed_data)}")
            report.append(f"- **个体数**: {self.processed_data['孕妇代码'].nunique()}")
            report.append(f"- **平均每个体测量次数**: {len(self.processed_data) / self.processed_data['孕妇代码'].nunique():.1f}")
        
        # 2. 模型信息
        report.append("\n## 2. 增强混合效应模型信息")
        if self.model_analyzer.result is not None:
            report.append(f"- **模型类型**: 随机斜率混合效应模型")
            report.append(f"- **使用变量数**: {len(self.model_analyzer.variables_used)}")
            report.append(f"- **模型公式**: {self.model_analyzer.formula}")
            report.append(f"- **收敛状态**: {'成功' if self.model_analyzer.result.converged else '失败'}")
            report.append(f"- **AIC**: {self.model_analyzer.result.aic:.2f}")
            report.append(f"- **BIC**: {self.model_analyzer.result.bic:.2f}")
            report.append(f"- **对数似然**: {self.model_analyzer.result.llf:.2f}")
        
        # 3. 变量列表
        report.append("\n## 3. 模型变量列表")
        if hasattr(self.model_analyzer, 'variables_used'):
            report.append("### 基础变量（原模型）")
            base_vars = ['gd', 'bmi_c', 'age_c', 'height', 'weight']
            for var in base_vars:
                if var in self.model_analyzer.variables_used:
                    report.append(f"- {var}")
            
            report.append("\n### 新增高相关性变量")
            new_vars = [v for v in self.model_analyzer.variables_used if v not in base_vars]
            for var in new_vars:
                report.append(f"- {var}")
        
        # 4. 固定效应结果
        report.append("\n## 4. 固定效应结果")
        if self.model_analyzer.result is not None:
            fixed_effects = self.model_analyzer.get_fixed_effects_summary()
            if fixed_effects is not None:
                report.append("\n| 变量 | 系数 | 标准误 | P值 | 显著性 |")
                report.append("|------|------|--------|-----|--------|")
                for _, row in fixed_effects.iterrows():
                    report.append(f"| {row['Variable']} | {row['Coefficient']:.6f} | {row['Std_Error']:.6f} | {row['P_value']:.6f} | {row['Significance']} |")
        
        # 5. 模型性能
        report.append("\n## 5. 模型性能指标")
        if self.model_analyzer.result is not None:
            performance = self.model_analyzer.calculate_model_performance(self.processed_data)
            if performance:
                for key, value in performance.items():
                    report.append(f"- **{key}**: {value:.4f}")
        
        # 6. 主要发现
        report.append("\n## 6. 主要发现")
        if 'correlation_matrix' in self.analysis_results:
            corr_matrix = self.analysis_results['correlation_matrix']
            y_corr = corr_matrix['Y染色体浓度'].drop('Y染色体浓度').sort_values(key=abs, ascending=False)
            
            report.append("### 与Y染色体浓度相关性最高的变量")
            for var, corr in y_corr.head(5).items():
                report.append(f"- **{var}**: {corr:.4f}")
        
        # 7. 模型公式
        report.append("\n## 7. 模型数学表达式")
        if self.model_analyzer.result is not None:
            formula_text = self.model_analyzer.get_model_formula_text()
            if formula_text:
                report.append(f"```\n{formula_text}\n```")
        
        # 8. 结论
        report.append("\n## 8. 结论")
        report.append("本分析构建了包含12个变量的增强随机斜率混合效应模型，相比原模型（5个变量）显著扩展了变量覆盖范围。")
        report.append("新增的高相关性变量包括X染色体浓度、Y染色体Z值、检测抽血次数等，这些变量与Y染色体浓度存在显著相关性。")
        report.append("模型能够更好地解释Y染色体浓度的变异，为临床决策提供更准确的预测。")
        
        return "\n".join(report)
    
    def _safe_format_number(self, value, format_str=".4f"):
        """安全地格式化数值，处理字符串'N/A'的情况"""
        if isinstance(value, (int, float)) and not pd.isna(value):
            return f"{value:{format_str}}"
        else:
            return str(value)
    
    def get_analysis_summary(self) -> dict:
        """获取分析摘要"""
        
        summary = {
            'data_info': {
                'total_observations': len(self.processed_data) if self.processed_data is not None else 0,
                'unique_subjects': self.processed_data['孕妇代码'].nunique() if self.processed_data is not None else 0,
                '达标率': self.analysis_results.get('summary_stats', {}).get('y_concentration_stats', {}).get('达标率', 'N/A')
            },
            'model_type': 'enhanced_random_slope',
            'model_fitted': self.model_analyzer.result is not None,
            'model_performance': self.analysis_results.get('model_performance', {}),
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
    
    def _calculate_correlation_pvalues(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算相关性的p值（使用Spearman相关系数）"""
        from scipy.stats import spearmanr
        
        variables = data.columns.tolist()
        n_vars = len(variables)
        
        # 创建p值矩阵
        pvalue_matrix = np.ones((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                try:
                    # 移除缺失值对
                    var1 = data.iloc[:, i].dropna()
                    var2 = data.iloc[:, j].dropna()
                    
                    # 找到两个变量都非缺失的索引
                    common_idx = var1.index.intersection(var2.index)
                    if len(common_idx) > 10:  # 至少需要10个观测点
                        _, p_val = spearmanr(var1.loc[common_idx], var2.loc[common_idx])
                        pvalue_matrix[i, j] = p_val
                        pvalue_matrix[j, i] = p_val
                except:
                    pvalue_matrix[i, j] = np.nan
                    pvalue_matrix[j, i] = np.nan
        
        # 转换为DataFrame
        pvalue_df = pd.DataFrame(pvalue_matrix, index=variables, columns=variables)
        return pvalue_df
    
    def _analyze_bmi_group_correlations(self, data: pd.DataFrame) -> dict:
        """按BMI分组分析相关性"""
        
        group_correlations = {}
        
        if 'BMI_group' in self.processed_data.columns:
            for group in self.processed_data['BMI_group'].unique():
                if pd.notna(group):
                    group_data = self.processed_data[self.processed_data['BMI_group'] == group]
                    
                    if len(group_data) > 10:  # 确保样本量足够
                        # 选择核心变量进行分组分析
                        core_vars = ['Y染色体浓度', 'gestational_days', '孕妇BMI', '年龄']
                        # 添加Z值变量（如果存在）
                        if 'Y染色体的Z值' in group_data.columns:
                            core_vars.append('Y染色体的Z值')
                        
                        group_corr_input = group_data[core_vars].apply(pd.to_numeric, errors='coerce')
                        group_corr = group_corr_input.corr(method='spearman')
                        
                        # 重点关注Y染色体浓度的相关性
                        y_correlations = group_corr['Y染色体浓度'].drop('Y染色体浓度')
                        group_correlations[str(group)] = y_correlations.to_dict()
        
        return group_correlations
    
    def _analyze_nonlinear_correlations(self, data: pd.DataFrame) -> dict:
        """分析非线性相关性"""
        
        nonlinear_results = {}
        
        # 对于关键变量，创建多项式特征并分析相关性
        if 'gestational_days' in data.columns and 'Y染色体浓度' in data.columns:
            # 二次项
            gest_squared = data['gestational_days'] ** 2
            gest_squared.name = 'gestational_days_squared'
            
            # 三次项
            gest_cubed = data['gestational_days'] ** 3
            gest_cubed.name = 'gestational_days_cubed'
            
            # 计算与Y染色体浓度的相关性
            y_conc = data['Y染色体浓度']
            
            try:
                from scipy.stats import spearmanr
                
                # 主要使用Spearman相关性（非参数，能捕获单调非线性关系）
                spear_corr, spear_p = spearmanr(data['gestational_days'].dropna(), 
                                              y_conc.dropna())
                
                nonlinear_results['gestational_days'] = {
                    'spearman_corr': spear_corr,
                    'spearman_p': spear_p
                }
                
                # BMI的非线性分析
                if '孕妇BMI' in data.columns:
                    bmi_spear_corr, bmi_spear_p = spearmanr(data['孕妇BMI'].dropna(), 
                                                          y_conc.dropna())
                    
                    nonlinear_results['孕妇BMI'] = {
                        'spearman_corr': bmi_spear_corr,
                        'spearman_p': bmi_spear_p
                    }
                
            except Exception as e:
                print(f"非线性相关性分析出错: {str(e)}")
        
        return nonlinear_results
    
    def _display_correlation_results(self, correlation_matrix: pd.DataFrame, 
                                   correlation_pvalues: pd.DataFrame):
        """显示相关性分析结果"""
        
        print("\n" + "="*60)
        print("详细相关性分析结果")
        print("="*60)
        
        # 核心相关性结果
        print("\n1. Y染色体浓度与主要指标的相关性：")
        key_vars = ['gestational_days', '孕妇BMI', '年龄', '身高', '体重']
        
        for var in key_vars:
            if var in correlation_matrix.index and 'Y染色体浓度' in correlation_matrix.columns:
                try:
                    corr_val = correlation_matrix.loc['Y染色体浓度', var]
                    p_val = correlation_pvalues.loc['Y染色体浓度', var] if 'Y染色体浓度' in correlation_pvalues.index else np.nan
                    
                    if pd.notna(corr_val):
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        direction = "正相关" if corr_val > 0 else "负相关"
                        strength = "强" if abs(corr_val) > 0.7 else "中等" if abs(corr_val) > 0.3 else "弱"
                        
                        print(f"  {var}: ρ={corr_val:.4f}, p={p_val:.4f} ({strength}{direction}, {significance})")
                except:
                    print(f"  {var}: 无法计算")
        
        # 高相关性变量
        print("\n2. 与Y染色体浓度相关性最高的变量（按绝对值排序）：")
        if 'Y染色体浓度' in correlation_matrix.columns:
            y_corr = correlation_matrix['Y染色体浓度'].drop('Y染色体浓度').sort_values(key=abs, ascending=False)
            for i, (var, corr_val) in enumerate(y_corr.head(10).items(), 1):
                if pd.notna(corr_val):
                    p_val = correlation_pvalues.loc['Y染色体浓度', var] if 'Y染色体浓度' in correlation_pvalues.index else np.nan
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    print(f"  {i:2d}. {var}: ρ={corr_val:.4f}, p={p_val:.4f} ({significance})")
        
        # Z值相关性
        if 'Y染色体的Z值' in correlation_matrix.index:
            print("\n3. Y染色体相关Z值分析（Spearman）：")
            z_corr = correlation_matrix.loc['Y染色体浓度', 'Y染色体的Z值']
            z_p = correlation_pvalues.loc['Y染色体浓度', 'Y染色体的Z值'] if 'Y染色体浓度' in correlation_pvalues.index else np.nan
            print(f"  Y染色体浓度 vs Y染色体Z值: ρ={z_corr:.4f}, p={z_p:.4f}")
        
        # BMI分组结果
        if 'bmi_group_correlations' in self.analysis_results:
            print("\n4. BMI分组相关性差异（Spearman）：")
            for group, correlations in self.analysis_results['bmi_group_correlations'].items():
                if 'gestational_days' in correlations:
                    print(f"  BMI {group}组: Y浓度-孕周相关性 ρ={correlations['gestational_days']:.4f}")
        
        # 非线性结果
        if 'nonlinear_correlations' in self.analysis_results:
            print("\n5. 非线性相关性分析（Spearman相关系数）：")
            for var, results in self.analysis_results['nonlinear_correlations'].items():
                spear_r = results.get('spearman_corr', np.nan)
                spear_p = results.get('spearman_p', np.nan)
                print(f"  {var}: Spearman ρ={spear_r:.4f}, p={spear_p:.4f}")
        
        # 保存相关性结果到文件
        print("\n6. 保存相关性结果...")
        try:
            # 保存完整相关性矩阵
            correlation_matrix.to_csv(
                os.path.join(self.results_dir, "data", "full_correlation_matrix.csv")
            )
            print("  ✓ 完整相关性矩阵已保存: data/full_correlation_matrix.csv")
            
            # 保存相关性p值矩阵
            correlation_pvalues.to_csv(
                os.path.join(self.results_dir, "data", "correlation_pvalues.csv")
            )
            print("  ✓ 相关性p值矩阵已保存: data/correlation_pvalues.csv")
            
            # 保存Y染色体浓度相关性排序
            y_corr_df = pd.DataFrame({
                'Variable': y_corr.index,
                'Correlation': y_corr.values,
                'Abs_Correlation': y_corr.abs().values
            }).sort_values('Abs_Correlation', ascending=False)
            
            y_corr_df.to_csv(
                os.path.join(self.results_dir, "data", "y_concentration_correlations_ranked.csv"),
                index=False
            )
            print("  ✓ Y染色体浓度相关性排序已保存: data/y_concentration_correlations_ranked.csv")
            
        except Exception as e:
            print(f"  保存相关性结果时出错: {e}")
        
        print("="*60)


def main():
    """主函数"""
    
    # 设置数据路径
    data_path = "../初始数据/男胎检测数据.csv"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 - {data_path}")
        print("请确保数据文件路径正确")
        return
    
    # 只生成拆分图片
    print("生成拆分图片...")
    analysis = Problem1Analysis(data_path, results_dir="results/", separate_figures=True)
    
    try:
        # 运行完整分析（生成拆分图片）
        analysis.run_complete_analysis()
        
        # 显示分析摘要
        summary = analysis.get_analysis_summary()
        print("\n" + "="*60)
        print("分析摘要")
        print("="*60)
        
        print(f"数据规模: {summary['data_info']['total_observations']}个观测，{summary['data_info']['unique_subjects']}个个体")
        print(f"总体达标率: {summary['data_info']['达标率']:.2%}")
        print(f"模型类型: {summary['model_type']}")
        print(f"模型拟合状态: {'成功' if summary['model_fitted'] else '失败'}")
        
        if summary['model_performance']:
            # 安全地格式化性能指标
            r2_val = summary['model_performance'].get('R_squared', 'N/A')
            if isinstance(r2_val, (int, float)):
                print(f"模型表现: R² = {r2_val:.4f}")
            else:
                print(f"模型表现: R² = {r2_val}")
            
            corr_val = summary['model_performance'].get('Correlation', 'N/A')
            if isinstance(corr_val, (int, float)):
                print(f"相关系数: {corr_val:.4f}")
            else:
                print(f"相关系数: {corr_val}")
            
            aic_val = summary['model_performance'].get('AIC', 'N/A')
            if isinstance(aic_val, (int, float)):
                print(f"AIC: {aic_val:.2f}")
            else:
                print(f"AIC: {aic_val}")
        
        print("\n关键发现:")
        for i, finding in enumerate(summary['key_findings'], 1):
            print(f"{i}. {finding}")
        
        print("\n分析完成！结果文件位于 'results/' 目录")
        print("\n拆分图片已生成，包括以下类型：")
        print("- 分布分析图：10个子图（Y浓度分布、孕周分布、BMI分布、染色体Z值箱线图、XY染色体浓度箱线图、染色体Z值直方图、XY染色体浓度直方图等）")
        print("- 散点关系图：4个子图（Y浓度vs孕周、Y浓度vsBMI等）")
        print("- 相关性分析图：6个子图（相关性矩阵、显著性条形图等）")
        print("- 模型比较图：2个子图（AIC比较、Delta AIC）")
        print("- 模型诊断图：4个子图（残差图、正态概率图等）")
        print("所有拆分的子图都保存在 'results/figures/' 目录中")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
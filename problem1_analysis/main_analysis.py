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
        
        # 第九步：保存关系模型
        print("\n步骤9: 保存关系模型")
        self.save_relationship_model()
        
        # 第十步：结果保存与报告生成
        print("\n步骤10: 结果保存与报告生成")
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
        
        
        print(f"共拟合了{len(self.model_analyzer.results)}个模型")
    
    def model_comparison(self):
        """模型比较与选择"""
        
        # 使用增强的模型比较（包含R方等统计指标）
        comparison_df = self.model_analyzer.compare_models_enhanced(self.processed_data)
        self.analysis_results['model_comparison'] = comparison_df
        
        # 绘制模型比较图
        if not comparison_df.empty:
            self.visualizer.plot_model_comparison(
                comparison_df,
                save_path=self.results_manager.get_figure_path("model_comparison.png"),
                separate_figs=self.separate_figures
            )
        
        print("模型比较完成")
        
        # 显示增强的模型比较结果
        print("\n" + "="*80)
        print("增强的模型比较结果（包含R方等统计指标）")
        print("="*80)
        print(comparison_df.round(4))
    
    def significance_testing(self):
        """显著性检验"""
        
        # 获取所有模型的固定效应和统计指标
        all_fixed_effects = {}
        all_model_stats = {}
        
        for model_name in self.model_analyzer.results.keys():
            print(f"分析模型: {model_name}")
            
            # 固定效应检验
            fixed_effects_df = self.model_analyzer.test_fixed_effects(model_name)
            if fixed_effects_df is not None:
                all_fixed_effects[model_name] = fixed_effects_df
            
            # 计算R平方等统计指标
            r_squared_info = self.model_analyzer.calculate_r_squared(
                model_name, 
                self.processed_data
            )
            if r_squared_info is not None:
                all_model_stats[model_name] = r_squared_info
                
                # 如果是最佳模型，进行额外分析
                if model_name == self.model_analyzer.best_model:
                    self.analysis_results['best_model_fixed_effects'] = fixed_effects_df
                    self.analysis_results['best_model_r_squared'] = r_squared_info
                    
                    # 随机效应分析
                    random_effects_info = self.model_analyzer.test_random_effects(model_name)
                    self.analysis_results['best_model_random_effects'] = random_effects_info
                    
                    # 绘制固定效应图
                    if fixed_effects_df is not None:
                        self.visualizer.plot_fixed_effects(
                            fixed_effects_df,
                            save_path=self.results_manager.get_figure_path("fixed_effects.png")
                        )
        
        # 保存所有模型的结果
        self.analysis_results['all_fixed_effects'] = all_fixed_effects
        self.analysis_results['all_model_stats'] = all_model_stats
        
        print("显著性检验完成")
        
        # 显示所有模型的统计指标
        print("\n" + "="*80)
        print("所有模型的统计指标对比")
        print("="*80)
        
        for model_name, stats in all_model_stats.items():
            print(f"\n模型: {model_name}")
            print(f"  伪R²: {stats.get('Pseudo_R2', 'N/A'):.4f}")
            print(f"  相关系数: {stats.get('Correlation', 'N/A'):.4f}")
            print(f"  相关系数R²: {stats.get('Correlation_R2', 'N/A'):.4f}")
        
        # 显示最佳模型结果
        best_model_name = self.model_analyzer.best_model
        if best_model_name and best_model_name in all_fixed_effects:
            print(f"\n" + "="*50)
            print(f"最佳模型 {best_model_name} 的详细结果:")
            print("="*50)
            
            # 显示统计指标
            if best_model_name in all_model_stats:
                stats = all_model_stats[best_model_name]
                print(f"伪R²: {stats.get('Pseudo_R2', 'N/A'):.4f}")
                print(f"相关系数: {stats.get('Correlation', 'N/A'):.4f}")
                print(f"相关系数R²: {stats.get('Correlation_R2', 'N/A'):.4f}")
            
            # 显示固定效应
            fixed_effects_df = all_fixed_effects[best_model_name]
            print(f"\n固定效应结果:")
            for _, row in fixed_effects_df.iterrows():
                if row['Variable'] != 'const':
                    print(f"  {row['Variable']}: 系数={row['Coefficient']:.6f}, "
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
                    save_path=self.results_manager.get_figure_path("model_diagnostics.png"),
                    separate_figs=self.separate_figures
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
    
    def print_relationship_models(self):
        """打印和确定关系模型"""
        
        print("\n" + "="*80)
        print("关系模型确定与打印")
        print("="*80)
        
        # 获取最佳模型
        best_model_name = self.model_analyzer.best_model
        
        if not best_model_name:
            print("没有找到最佳模型")
            return
        
        print(f"\n最佳模型: {best_model_name}")
        print("-" * 50)
        
        # 获取模型结果
        if best_model_name in self.model_analyzer.results:
            model_result = self.model_analyzer.results[best_model_name]
            
            # 打印模型摘要
            print("\n1. 模型摘要:")
            print(model_result.summary())
            
            # 打印固定效应系数
            print("\n2. 固定效应系数:")
            print("-" * 30)
            fixed_effects = model_result.params
            for param, value in fixed_effects.items():
                print(f"{param:20s}: {value:10.6f}")
            
            # 打印随机效应方差
            print("\n3. 随机效应方差:")
            print("-" * 30)
            if hasattr(model_result, 'cov_re'):
                print("随机效应协方差矩阵:")
                print(model_result.cov_re)
            
            # 打印模型拟合统计
            print("\n4. 模型拟合统计:")
            print("-" * 30)
            print(f"AIC: {model_result.aic:.4f}")
            print(f"BIC: {model_result.bic:.4f}")
            print(f"对数似然: {model_result.llf:.4f}")
            
            # 打印模型公式
            print("\n5. 模型数学表达式:")
            print("-" * 30)
            self._print_model_formula(best_model_name, model_result)
            
            # 打印变量关系解释
            print("\n6. 变量关系解释:")
            print("-" * 30)
            self._print_variable_relationships(model_result)
            
        else:
            print(f"未找到模型 {best_model_name} 的结果")
    
    def _print_model_formula(self, model_name: str, model_result):
        """打印模型数学表达式"""
        
        # 获取固定效应系数
        params = model_result.params
        
        # 构建模型公式
        formula_parts = []
        
        # 截距项
        if 'const' in params:
            intercept = params['const']
            formula_parts.append(f"{intercept:.6f}")
        
        # 其他固定效应
        for param, coef in params.items():
            if param != 'const':
                if coef >= 0:
                    formula_parts.append(f"+ {coef:.6f} * {param}")
                else:
                    formula_parts.append(f"- {abs(coef):.6f} * {param}")
        
        # 构建完整公式
        formula = "Y染色体浓度 = " + " ".join(formula_parts)
        
        print(f"模型名称: {model_name}")
        print(f"数学表达式: {formula}")
        
        # 添加随机效应说明
        if 'random_intercept' in model_name or 'random_slope' in model_name:
            print("\n随机效应:")
            if 'random_intercept' in model_name:
                print("- 随机截距: 每个孕妇有不同的基线Y染色体浓度")
            if 'random_slope' in model_name:
                print("- 随机斜率: 每个孕妇的孕周效应不同")
    
    def _print_variable_relationships(self, model_result):
        """打印变量关系解释"""
        
        params = model_result.params
        pvalues = model_result.pvalues
        
        print("变量对Y染色体浓度的影响:")
        print()
        
        for param, coef in params.items():
            if param == 'const':
                continue
                
            pval = pvalues.get(param, np.nan)
            
            # 解释系数含义
            if param == 'gestational_days':
                print(f"• 孕周天数 (gestational_days):")
                print(f"  系数: {coef:.6f}")
                print(f"  含义: 孕周每增加1天，Y染色体浓度平均变化 {coef:.6f}")
                if not np.isnan(pval):
                    significance = "显著" if pval < 0.05 else "不显著"
                    print(f"  显著性: {significance} (p = {pval:.4f})")
                print()
                
            elif param == 'BMI_centered':
                print(f"• BMI (BMI_centered):")
                print(f"  系数: {coef:.6f}")
                print(f"  含义: BMI每增加1个单位，Y染色体浓度平均变化 {coef:.6f}")
                if not np.isnan(pval):
                    significance = "显著" if pval < 0.05 else "不显著"
                    print(f"  显著性: {significance} (p = {pval:.4f})")
                print()
                
            elif param == 'age_centered':
                print(f"• 年龄 (age_centered):")
                print(f"  系数: {coef:.6f}")
                print(f"  含义: 年龄每增加1岁，Y染色体浓度平均变化 {coef:.6f}")
                if not np.isnan(pval):
                    significance = "显著" if pval < 0.05 else "不显著"
                    print(f"  显著性: {significance} (p = {pval:.4f})")
                print()
                
            elif param == '身高':
                print(f"• 身高:")
                print(f"  系数: {coef:.6f}")
                print(f"  含义: 身高每增加1cm，Y染色体浓度平均变化 {coef:.6f}")
                if not np.isnan(pval):
                    significance = "显著" if pval < 0.05 else "不显著"
                    print(f"  显著性: {significance} (p = {pval:.4f})")
                print()
                
            elif param == '体重':
                print(f"• 体重:")
                print(f"  系数: {coef:.6f}")
                print(f"  含义: 体重每增加1kg，Y染色体浓度平均变化 {coef:.6f}")
                if not np.isnan(pval):
                    significance = "显著" if pval < 0.05 else "不显著"
                    print(f"  显著性: {significance} (p = {pval:.4f})")
                print()
    
    def save_relationship_model(self):
        """保存关系模型到文件"""
        
        best_model_name = self.model_analyzer.best_model
        
        if not best_model_name or best_model_name not in self.model_analyzer.results:
            print("没有找到最佳模型，无法保存关系模型")
            return
        
        model_result = self.model_analyzer.results[best_model_name]
        
        # 创建模型报告文件
        model_report_path = os.path.join(self.results_dir, "reports", "relationship_model_report.md")
        os.makedirs(os.path.dirname(model_report_path), exist_ok=True)
        
        with open(model_report_path, 'w', encoding='utf-8') as f:
            f.write("# 关系模型报告\n\n")
            f.write(f"**最佳模型**: {best_model_name}\n\n")
            
            # 模型摘要
            f.write("## 1. 模型摘要\n\n")
            f.write("```\n")
            f.write(str(model_result.summary()))
            f.write("\n```\n\n")
            
            # 固定效应系数
            f.write("## 2. 固定效应系数\n\n")
            f.write("| 变量 | 系数 | 标准误 | P值 |\n")
            f.write("|------|------|--------|-----|\n")
            
            params = model_result.params
            pvalues = model_result.pvalues
            std_errors = model_result.bse
            
            for param in params.index:
                coef = params[param]
                pval = pvalues.get(param, np.nan)
                std_err = std_errors.get(param, np.nan)
                significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                f.write(f"| {param} | {coef:.6f} | {std_err:.6f} | {pval:.4f}{significance} |\n")
            
            f.write("\n")
            
            # 模型数学表达式
            f.write("## 3. 模型数学表达式\n\n")
            formula_parts = []
            if 'const' in params:
                intercept = params['const']
                formula_parts.append(f"{intercept:.6f}")
            
            for param, coef in params.items():
                if param != 'const':
                    if coef >= 0:
                        formula_parts.append(f"+ {coef:.6f} * {param}")
                    else:
                        formula_parts.append(f"- {abs(coef):.6f} * {param}")
            
            formula = "Y染色体浓度 = " + " ".join(formula_parts)
            f.write(f"```\n{formula}\n```\n\n")
            
            # 变量关系解释
            f.write("## 4. 变量关系解释\n\n")
            for param, coef in params.items():
                if param == 'const':
                    continue
                    
                pval = pvalues.get(param, np.nan)
                significance = "显著" if not np.isnan(pval) and pval < 0.05 else "不显著"
                
                if param == 'gestational_days':
                    f.write(f"### 孕周天数 (gestational_days)\n")
                    f.write(f"- **系数**: {coef:.6f}\n")
                    f.write(f"- **含义**: 孕周每增加1天，Y染色体浓度平均变化 {coef:.6f}\n")
                    f.write(f"- **显著性**: {significance} (p = {pval:.4f})\n\n")
                    
                elif param == 'BMI_centered':
                    f.write(f"### BMI (BMI_centered)\n")
                    f.write(f"- **系数**: {coef:.6f}\n")
                    f.write(f"- **含义**: BMI每增加1个单位，Y染色体浓度平均变化 {coef:.6f}\n")
                    f.write(f"- **显著性**: {significance} (p = {pval:.4f})\n\n")
                    
                elif param == 'age_centered':
                    f.write(f"### 年龄 (age_centered)\n")
                    f.write(f"- **系数**: {coef:.6f}\n")
                    f.write(f"- **含义**: 年龄每增加1岁，Y染色体浓度平均变化 {coef:.6f}\n")
                    f.write(f"- **显著性**: {significance} (p = {pval:.4f})\n\n")
                    
                elif param == '身高':
                    f.write(f"### 身高\n")
                    f.write(f"- **系数**: {coef:.6f}\n")
                    f.write(f"- **含义**: 身高每增加1cm，Y染色体浓度平均变化 {coef:.6f}\n")
                    f.write(f"- **显著性**: {significance} (p = {pval:.4f})\n\n")
                    
                elif param == '体重':
                    f.write(f"### 体重\n")
                    f.write(f"- **系数**: {coef:.6f}\n")
                    f.write(f"- **含义**: 体重每增加1kg，Y染色体浓度平均变化 {coef:.6f}\n")
                    f.write(f"- **显著性**: {significance} (p = {pval:.4f})\n\n")
            
            # 模型拟合统计
            f.write("## 5. 模型拟合统计\n\n")
            f.write(f"- **AIC**: {model_result.aic:.4f}\n")
            f.write(f"- **BIC**: {model_result.bic:.4f}\n")
            f.write(f"- **对数似然**: {model_result.llf:.4f}\n\n")
            
            # 随机效应说明
            if 'random_intercept' in best_model_name or 'random_slope' in best_model_name:
                f.write("## 6. 随机效应说明\n\n")
                if 'random_intercept' in best_model_name:
                    f.write("- **随机截距**: 每个孕妇有不同的基线Y染色体浓度\n")
                if 'random_slope' in best_model_name:
                    f.write("- **随机斜率**: 每个孕妇的孕周效应不同\n")
        
        print(f"关系模型报告已保存到: {model_report_path}")
    
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
        
        # 保存增强的相关性分析结果
        if all(key in self.analysis_results for key in ['correlation_matrix', 'correlation_pvalues', 'bmi_group_correlations', 'nonlinear_correlations']):
            self.results_manager.save_enhanced_correlation_results(
                self.analysis_results['correlation_matrix'],
                self.analysis_results['correlation_pvalues'],
                self.analysis_results['bmi_group_correlations'],
                self.analysis_results['nonlinear_correlations']
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
        
        # 保存所有模型的固定效应结果
        if 'all_fixed_effects' in self.analysis_results:
            self.results_manager.save_fixed_effects_results(self.analysis_results['all_fixed_effects'])
        elif 'best_model_fixed_effects' in self.analysis_results:
            # 兼容性：如果只有最佳模型结果，也保存
            fixed_effects_dict = {
                self.model_analyzer.best_model: self.analysis_results['best_model_fixed_effects']
            }
            self.results_manager.save_fixed_effects_results(fixed_effects_dict)
        
        # 保存所有模型的统计指标
        if 'all_model_stats' in self.analysis_results:
            self.results_manager.save_model_statistics(self.analysis_results['all_model_stats'])
        
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
        
        print("="*60)
        print("详细相关性分析结果")
        print("="*60)
        
        # 核心相关性结果
        print("\\n1. Y染色体浓度与主要指标的相关性：")
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
        
        # Z值相关性
        if 'Y染色体的Z值' in correlation_matrix.index:
            print("\n2. Y染色体相关Z值分析（Spearman）：")
            z_corr = correlation_matrix.loc['Y染色体浓度', 'Y染色体的Z值']
            z_p = correlation_pvalues.loc['Y染色体浓度', 'Y染色体的Z值'] if 'Y染色体浓度' in correlation_pvalues.index else np.nan
            print(f"  Y染色体浓度 vs Y染色体Z值: ρ={z_corr:.4f}, p={z_p:.4f}")
        
        # BMI分组结果
        if 'bmi_group_correlations' in self.analysis_results:
            print("\n3. BMI分组相关性差异（Spearman）：")
            for group, correlations in self.analysis_results['bmi_group_correlations'].items():
                if 'gestational_days' in correlations:
                    print(f"  BMI {group}组: Y浓度-孕周相关性 ρ={correlations['gestational_days']:.4f}")
        
        # 非线性结果
        if 'nonlinear_correlations' in self.analysis_results:
            print("\n4. 非线性相关性分析（Spearman相关系数）：")
            for var, results in self.analysis_results['nonlinear_correlations'].items():
                spear_r = results.get('spearman_corr', np.nan)
                spear_p = results.get('spearman_p', np.nan)
                print(f"  {var}: Spearman ρ={spear_r:.4f}, p={spear_p:.4f}")


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
        print(f"最佳模型: {summary['best_model']}")
        
        if summary['model_performance']:
            print(f"模型表现: R² = {summary['model_performance'].get('Pseudo_R2', 'N/A'):.4f}")
        
        print("\n关键发现:")
        for i, finding in enumerate(summary['key_findings'], 1):
            print(f"{i}. {finding}")
        
        print("\n分析完成！结果文件位于 'results/' 目录")
        print("\n拆分图片已生成，包括以下类型：")
        print("- 分布分析图：6个子图（Y浓度分布、孕周分布、BMI分布等）")
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
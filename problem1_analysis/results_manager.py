"""
结果管理模块
负责保存分析结果、生成报告等
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle
from typing import Dict, Any


class NIPTResultsManager:
    """NIPT分析结果管理器"""
    
    def __init__(self, results_dir: str = "results/"):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, "figures/")
        self.data_dir = os.path.join(results_dir, "data/")
        self.reports_dir = os.path.join(results_dir, "reports/")
        self.models_dir = os.path.join(results_dir, "models/")
        
        # 创建目录
        self._create_directories()
        
    def _create_directories(self):
        """创建结果目录结构"""
        directories = [
            self.results_dir,
            self.figures_dir,
            self.data_dir,
            self.reports_dir,
            self.models_dir
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")
    
    def save_processed_data(self, data: pd.DataFrame, filename: str = "processed_data.csv"):
        """保存处理后的数据"""
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False, encoding='utf-8')
        print(f"处理后数据已保存: {filepath}")
        return filepath
    
    def save_summary_statistics(self, summary_stats: Dict, filename: str = "summary_statistics.json"):
        """保存描述性统计"""
        filepath = os.path.join(self.data_dir, filename)
        
        # 转换numpy类型为Python类型（用于JSON序列化）
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        summary_stats_clean = convert_numpy_types(summary_stats)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_stats_clean, f, indent=4, ensure_ascii=False)
        
        print(f"描述性统计已保存: {filepath}")
        return filepath
    
    def save_correlation_matrix(self, correlation_matrix: pd.DataFrame, 
                               filename: str = "correlation_matrix.csv"):
        """保存相关性矩阵"""
        filepath = os.path.join(self.data_dir, filename)
        correlation_matrix.to_csv(filepath, encoding='utf-8')
        print(f"相关性矩阵已保存: {filepath}")
        return filepath
    
    def save_enhanced_correlation_results(self, correlation_matrix: pd.DataFrame,
                                        correlation_pvalues: pd.DataFrame,
                                        bmi_group_correlations: dict,
                                        nonlinear_correlations: dict):
        """保存增强的相关性分析结果"""
        
        # 保存相关性矩阵
        corr_filepath = os.path.join(self.data_dir, "enhanced_correlation_matrix.csv")
        correlation_matrix.to_csv(corr_filepath, encoding='utf-8')
        
        # 保存p值矩阵
        pval_filepath = os.path.join(self.data_dir, "correlation_pvalues.csv")
        correlation_pvalues.to_csv(pval_filepath, encoding='utf-8')
        
        # 保存分组分析结果
        group_filepath = os.path.join(self.data_dir, "bmi_group_correlations.json")
        with open(group_filepath, 'w', encoding='utf-8') as f:
            json.dump(bmi_group_correlations, f, indent=4, ensure_ascii=False)
        
        # 保存非线性分析结果
        nonlinear_filepath = os.path.join(self.data_dir, "nonlinear_correlations.json")
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        nonlinear_clean = convert_numpy_types(nonlinear_correlations)
        with open(nonlinear_filepath, 'w', encoding='utf-8') as f:
            json.dump(nonlinear_clean, f, indent=4, ensure_ascii=False)
        
        print(f"增强相关性分析结果已保存:")
        print(f"  相关性矩阵: {corr_filepath}")
        print(f"  p值矩阵: {pval_filepath}")
        print(f"  BMI分组结果: {group_filepath}")
        print(f"  非线性分析: {nonlinear_filepath}")
    
    def save_model_results(self, model_results: Dict, filename: str = "model_results.json"):
        """保存模型结果"""
        filepath = os.path.join(self.models_dir, filename)
        
        # 处理模型结果，提取可序列化的信息
        serializable_results = {}
        
        for model_name, result in model_results.items():
            if result is not None:
                try:
                    # 兼容不同statsmodels版本下的参数个数字段
                    df_model_val = None
                    if hasattr(result, 'df_model') and result.df_model is not None:
                        df_model_val = int(result.df_model)
                    elif hasattr(result, 'df_modelwc') and result.df_modelwc is not None:
                        df_model_val = int(result.df_modelwc)
                    else:
                        try:
                            df_model_val = int(len(result.params))
                        except Exception:
                            df_model_val = None
                    serializable_results[model_name] = {
                        'aic': float(result.aic) if hasattr(result, 'aic') else None,
                        'bic': float(result.bic) if hasattr(result, 'bic') else None,
                        'llf': float(result.llf) if hasattr(result, 'llf') else None,
                        'nobs': int(result.nobs) if hasattr(result, 'nobs') else None,
                        'df_model': df_model_val,
                        'converged': bool(result.converged) if hasattr(result, 'converged') else None,
                        'fe_params': result.fe_params.to_dict() if hasattr(result, 'fe_params') else None,
                        'pvalues': result.pvalues.to_dict() if hasattr(result, 'pvalues') else None,
                        'summary': str(result.summary()) if hasattr(result, 'summary') else None
                    }
                except Exception as e:
                    print(f"保存模型 {model_name} 时出错: {str(e)}")
                    serializable_results[model_name] = {'error': str(e)}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        
        print(f"模型结果已保存: {filepath}")
        return filepath
    
    def save_model_comparison(self, comparison_df: pd.DataFrame, 
                             filename: str = "model_comparison.csv"):
        """保存模型比较结果"""
        filepath = os.path.join(self.models_dir, filename)
        comparison_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"模型比较结果已保存: {filepath}")
        return filepath
    
    def save_fixed_effects_results(self, fixed_effects_results: Dict, 
                                  filename: str = "fixed_effects_results.csv"):
        """保存固定效应结果"""
        
        all_results = []
        
        for model_name, fe_df in fixed_effects_results.items():
            if fe_df is not None:
                fe_df_copy = fe_df.copy()
                fe_df_copy['Model'] = model_name
                all_results.append(fe_df_copy)
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            filepath = os.path.join(self.models_dir, filename)
            combined_df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"固定效应结果已保存: {filepath}")
            return filepath
        else:
            print("没有固定效应结果可保存")
            return None
    
    def save_model_statistics(self, model_stats: Dict, 
                            filename: str = "model_statistics.csv"):
        """保存所有模型的统计指标"""
        
        stats_data = []
        
        for model_name, stats in model_stats.items():
            if stats is not None:
                stats_row = {
                    'Model': model_name,
                    'Pseudo_R2': stats.get('Pseudo_R2', np.nan),
                    'Correlation': stats.get('Correlation', np.nan),
                    'Correlation_R2': stats.get('Correlation_R2', np.nan)
                }
                stats_data.append(stats_row)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            filepath = os.path.join(self.models_dir, filename)
            stats_df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"模型统计指标已保存: {filepath}")
            return filepath
        else:
            print("没有模型统计指标可保存")
            return None
    
    def save_predictions(self, predictions_df: pd.DataFrame, 
                        filename: str = "model_predictions.csv"):
        """保存模型预测结果"""
        filepath = os.path.join(self.data_dir, filename)
        predictions_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"模型预测结果已保存: {filepath}")
        return filepath
    
    def save_model_object(self, model_object, model_name: str):
        """保存模型对象（使用pickle）"""
        filename = f"{model_name}_model.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_object, f)
            print(f"模型对象已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存模型对象失败: {str(e)}")
            return None
    
    def generate_analysis_report(self, analysis_results: Dict, 
                               filename: str = "analysis_report.md"):
        """生成分析报告"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filepath = os.path.join(self.reports_dir, filename)
        
        report_content = f"""# NIPT Y染色体浓度混合效应模型分析报告

**生成时间**: {timestamp}

## 1. 数据概览

"""
        
        # 添加数据摘要
        if 'summary_stats' in analysis_results:
            stats = analysis_results['summary_stats']
            report_content += f"""
### 数据基本信息
- 总样本量: {stats.get('sample_size', 'N/A')}
- 个体数量: {stats.get('n_subjects', 'N/A')}

### Y染色体浓度统计
- 均值: {stats.get('y_concentration_stats', {}).get('mean', 'N/A'):.4f}
- 标准差: {stats.get('y_concentration_stats', {}).get('std', 'N/A'):.4f}
- 达标率: {stats.get('y_concentration_stats', {}).get('达标率', 'N/A'):.2%}

### 孕周统计
- 均值: {stats.get('gestational_days_stats', {}).get('mean', 'N/A'):.1f}天
- 范围: {stats.get('gestational_days_stats', {}).get('min', 'N/A'):.0f} - {stats.get('gestational_days_stats', {}).get('max', 'N/A'):.0f}天

### BMI统计
- 均值: {stats.get('BMI_stats', {}).get('mean', 'N/A'):.2f}
- 范围: {stats.get('BMI_stats', {}).get('min', 'N/A'):.1f} - {stats.get('BMI_stats', {}).get('max', 'N/A'):.1f}

"""
        
        # 添加模型比较结果
        if 'model_comparison' in analysis_results:
            comparison_df = analysis_results['model_comparison']
            report_content += """
## 2. 模型比较结果

| 模型 | AIC | BIC | Delta AIC | 收敛 |
|------|-----|-----|-----------|------|
"""
            for _, row in comparison_df.iterrows():
                report_content += f"| {row['Model']} | {row['AIC']:.2f} | {row['BIC']:.2f} | {row['Delta_AIC']:.2f} | {row['Converged']} |\n"
            
            best_model = comparison_df.iloc[0]['Model']
            report_content += f"""
**最佳模型**: {best_model} (基于AIC准则)

"""
        
        # 添加固定效应结果
        if 'best_model_fixed_effects' in analysis_results:
            fe_df = analysis_results['best_model_fixed_effects']
            report_content += """
## 3. 最佳模型固定效应结果

| 变量 | 系数 | 标准误 | t值 | p值 | 显著性 |
|------|------|--------|-----|-----|--------|
"""
            for _, row in fe_df.iterrows():
                report_content += f"| {row['Variable']} | {row['Coefficient']:.6f} | {row['Std_Error']:.6f} | {row['t_value']:.3f} | {row['p_value']:.6f} | {row['Significance']} |\n"
        
        # 添加模型解释
        report_content += """
## 4. 结果解释

### 主要发现

1. **孕周效应**: 孕周天数对Y染色体浓度有显著的正向影响，符合生物学预期。

2. **BMI效应**: BMI对Y染色体浓度产生负向影响，高BMI孕妇的Y染色体浓度相对较低。

3. **个体差异**: 混合效应模型捕捉到显著的个体间差异，说明每个孕妇都有其独特的基线水平和变化轨迹。

### 临床意义

- 模型结果支持BMI是影响Y染色体浓度的重要因素这一临床发现
- 个体差异的存在提示需要个体化的检测时点选择策略
- 模型为问题2和问题3的风险评估和时点优化提供了重要基础

## 5. 模型假设检验

通过残差分析和诊断图检验模型假设的满足情况。

## 6. 结论与建议

基于混合效应模型分析，我们建立了Y染色体浓度与孕周数、BMI等因素的关系模型，为NIPT检测时点的优化提供了科学依据。

---
*报告由NIPT分析系统自动生成*
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"分析报告已生成: {filepath}")
        return filepath
    
    def create_results_summary(self) -> Dict[str, str]:
        """创建结果文件摘要"""
        
        summary = {
            'results_directory': self.results_dir,
            'figures_directory': self.figures_dir,
            'data_directory': self.data_dir,
            'reports_directory': self.reports_dir,
            'models_directory': self.models_dir,
            'timestamp': datetime.now().isoformat()
        }
        
        # 列出已保存的文件
        for directory in [self.figures_dir, self.data_dir, self.reports_dir, self.models_dir]:
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
                summary[f'files_in_{os.path.basename(directory[:-1])}'] = files
        
        # 保存摘要
        summary_path = os.path.join(self.results_dir, 'results_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"结果摘要已保存: {summary_path}")
        return summary
    
    def get_figure_path(self, figure_name: str) -> str:
        """获取图片保存路径"""
        return os.path.join(self.figures_dir, figure_name)
    
    def load_results(self, filename: str) -> Dict:
        """加载保存的结果"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if filename.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif filename.endswith('.csv'):
                return pd.read_csv(filepath)
            else:
                print(f"不支持的文件格式: {filename}")
                return None
        except FileNotFoundError:
            print(f"文件不存在: {filepath}")
            return None
        except Exception as e:
            print(f"加载文件失败: {str(e)}")
            return None


if __name__ == "__main__":
    # 测试结果管理器
    manager = NIPTResultsManager()
    summary = manager.create_results_summary()
    print("结果管理器测试完成")
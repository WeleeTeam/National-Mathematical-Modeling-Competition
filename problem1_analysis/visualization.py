"""
可视化模块
生成各种分析图表和诊断图
"""

import matplotlib
# 设置非交互式后端，避免显示问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8')
except:
    # 如果旧版本seaborn，使用备选样式
    sns.set_style("whitegrid")


class NIPTVisualizer:
    """NIPT数据可视化器"""
    
    def __init__(self, output_dir: str = "results/figures/"):
        self.output_dir = output_dir
        
    def plot_individual_trajectories(self, data: pd.DataFrame, n_subjects: int = 20, 
                                   save_path: str = None):
        """绘制个体轨迹图"""
        
        plt.figure(figsize=(14, 10))
        
        # 随机选择n_subjects个个体
        unique_subjects = data['孕妇代码'].unique()
        selected_subjects = np.random.choice(unique_subjects, 
                                           min(n_subjects, len(unique_subjects)), 
                                           replace=False)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_subjects)))
        
        for i, subject in enumerate(selected_subjects):
            subject_data = data[data['孕妇代码'] == subject].sort_values('gestational_days')
            
            if len(subject_data) > 1:  # 只显示有多次测量的个体
                plt.plot(subject_data['gestational_days'], 
                        subject_data['Y染色体浓度'],
                        'o-', color=colors[i], alpha=0.7, linewidth=2,
                        markersize=6, label=f'孕妇{subject}')
        
        # 添加达标线
        plt.axhline(y=0.04, color='red', linestyle='--', linewidth=2, 
                   label='达标阈值(4%)')
        
        plt.xlabel('孕周(天数)', fontsize=12)
        plt.ylabel('Y染色体浓度', fontsize=12)
        plt.title('个体Y染色体浓度变化轨迹', fontsize=14, fontweight='bold')
        
        # 添加图例（只显示前10个）
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 11:  # 10个个体 + 达标线
            handles = handles[-1:] + handles[:10]  # 达标线 + 前10个个体
            labels = labels[-1:] + labels[:10]
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"个体轨迹图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, save_path: str = None):
        """绘制相关性热力图"""
        
        # 选择数值变量，优先使用数值化的计数列
        numeric_vars = ['Y染色体浓度', 'gestational_days', '孕妇BMI', '年龄', '身高', '体重']
        for optional_col in ['怀孕次数_num', '生产次数_num', '检测抽血次数_num']:
            if optional_col in data.columns:
                numeric_vars.append(optional_col)

        corr_input = data[numeric_vars].apply(pd.to_numeric, errors='coerce')
        correlation_data = corr_input.corr()
        
        plt.figure(figsize=(10, 8))
        
        # 创建热力图
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('变量间相关性热力图', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性热力图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def plot_scatter_relationships(self, data: pd.DataFrame, save_path: str = None):
        """绘制散点图矩阵"""
        
        # 选择主要变量
        vars_to_plot = ['Y染色体浓度', 'gestational_days', '孕妇BMI', '年龄']
        plot_data = data[vars_to_plot]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Y浓度 vs 孕周
        axes[0, 0].scatter(data['gestational_days'], data['Y染色体浓度'], 
                          alpha=0.6, c=data['孕妇BMI'], cmap='viridis')
        axes[0, 0].set_xlabel('孕周(天数)')
        axes[0, 0].set_ylabel('Y染色体浓度')
        axes[0, 0].set_title('Y浓度 vs 孕周数')
        axes[0, 0].axhline(y=0.04, color='red', linestyle='--', alpha=0.7)
        
        # Y浓度 vs BMI
        axes[0, 1].scatter(data['孕妇BMI'], data['Y染色体浓度'], 
                          alpha=0.6, c=data['gestational_days'], cmap='plasma')
        axes[0, 1].set_xlabel('BMI')
        axes[0, 1].set_ylabel('Y染色体浓度')
        axes[0, 1].set_title('Y浓度 vs BMI')
        axes[0, 1].axhline(y=0.04, color='red', linestyle='--', alpha=0.7)
        
        # Y浓度 vs 年龄
        axes[1, 0].scatter(data['年龄'], data['Y染色体浓度'], 
                          alpha=0.6, c=data['孕妇BMI'], cmap='coolwarm')
        axes[1, 0].set_xlabel('年龄')
        axes[1, 0].set_ylabel('Y染色体浓度')
        axes[1, 0].set_title('Y浓度 vs 年龄')
        axes[1, 0].axhline(y=0.04, color='red', linestyle='--', alpha=0.7)
        
        # BMI vs 孕周（展示个体差异）
        for subject in data['孕妇代码'].unique()[:20]:  # 只显示前20个个体
            subject_data = data[data['孕妇代码'] == subject]
            if len(subject_data) > 1:
                axes[1, 1].plot(subject_data['gestational_days'], 
                               subject_data['孕妇BMI'], 'o-', alpha=0.5)
        axes[1, 1].set_xlabel('孕周(天数)')
        axes[1, 1].set_ylabel('BMI')
        axes[1, 1].set_title('BMI随孕周变化(个体轨迹)')
        
        plt.suptitle('主要变量关系散点图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"散点图矩阵已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def plot_distribution_analysis(self, data: pd.DataFrame, save_path: str = None):
        """绘制分布分析图"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Y染色体浓度分布
        axes[0, 0].hist(data['Y染色体浓度'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=0.04, color='red', linestyle='--', linewidth=2, label='达标线')
        axes[0, 0].set_xlabel('Y染色体浓度')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('Y染色体浓度分布')
        axes[0, 0].legend()
        
        # 孕周分布
        axes[0, 1].hist(data['gestational_days'], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('孕周(天数)')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('孕周分布')
        
        # BMI分布
        axes[0, 2].hist(data['孕妇BMI'], bins=25, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('BMI')
        axes[0, 2].set_ylabel('频数')
        axes[0, 2].set_title('BMI分布')
        
        # Y浓度的正态概率图
        stats.probplot(data['Y染色体浓度'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Y浓度正态概率图')
        
        # 对数变换后的Y浓度分布
        log_y = np.log(data['Y染色体浓度'] + 0.001)
        axes[1, 1].hist(log_y, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('log(Y染色体浓度)')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('对数变换Y浓度分布')
        
        # 箱线图：不同BMI组的Y浓度
        data['BMI_group'] = pd.cut(data['孕妇BMI'], bins=4, labels=['低', '中低', '中高', '高'])
        sns.boxplot(data=data, x='BMI_group', y='Y染色体浓度', ax=axes[1, 2])
        axes[1, 2].axhline(y=0.04, color='red', linestyle='--', alpha=0.7)
        axes[1, 2].set_title('不同BMI组Y浓度分布')
        
        plt.suptitle('数据分布分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分布分析图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def plot_model_diagnostics(self, diagnostics: dict, model_name: str, save_path: str = None):
        """绘制模型诊断图"""
        
        residuals = diagnostics['residuals']
        fitted_values = diagnostics['fitted_values']
        std_residuals = diagnostics['std_residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 残差 vs 拟合值
        axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('拟合值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差 vs 拟合值')
        
        # 标准化残差 vs 拟合值
        axes[0, 1].scatter(fitted_values, std_residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].axhline(y=2, color='red', linestyle=':', alpha=0.5)
        axes[0, 1].axhline(y=-2, color='red', linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel('拟合值')
        axes[0, 1].set_ylabel('标准化残差')
        axes[0, 1].set_title('标准化残差 vs 拟合值')
        
        # 残差正态概率图
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('残差正态概率图')
        
        # 残差直方图
        axes[1, 1].hist(residuals, bins=25, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 1].set_xlabel('残差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('残差分布')
        
        # 添加正态性检验结果
        if 'shapiro_p_value' in diagnostics and not np.isnan(diagnostics['shapiro_p_value']):
            axes[1, 1].text(0.05, 0.95, f"Shapiro-Wilk p={diagnostics['shapiro_p_value']:.4f}", 
                           transform=axes[1, 1].transAxes, verticalalignment='top')
        
        plt.suptitle(f'模型诊断图 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型诊断图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def plot_predicted_vs_observed(self, predictions_df: pd.DataFrame, save_path: str = None):
        """绘制预测值vs观测值图"""
        
        plt.figure(figsize=(10, 8))
        
        # 散点图
        plt.scatter(predictions_df['observed'], predictions_df['predicted'], 
                   alpha=0.6, c=predictions_df['BMI'], cmap='viridis')
        
        # 添加对角线（完美预测线）
        min_val = min(predictions_df['observed'].min(), predictions_df['predicted'].min())
        max_val = max(predictions_df['observed'].max(), predictions_df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        
        # 添加达标线
        plt.axhline(y=0.04, color='orange', linestyle=':', alpha=0.7, label='预测达标线')
        plt.axvline(x=0.04, color='orange', linestyle=':', alpha=0.7, label='观测达标线')
        
        plt.xlabel('观测值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title('预测值 vs 观测值', fontsize=14, fontweight='bold')
        plt.colorbar(label='BMI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation = np.corrcoef(predictions_df['observed'], predictions_df['predicted'])[0, 1]
        plt.text(0.05, 0.95, f'R = {correlation:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测vs观测图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def plot_fixed_effects(self, fixed_effects_df: pd.DataFrame, save_path: str = None):
        """绘制固定效应系数图"""
        
        # 移除截距项（通常很大，影响可视化）
        plot_data = fixed_effects_df[fixed_effects_df['Variable'] != 'const'].copy()
        
        plt.figure(figsize=(10, 6))
        
        # 系数及置信区间
        x_pos = range(len(plot_data))
        plt.errorbar(x_pos, plot_data['Coefficient'], 
                    yerr=[plot_data['Coefficient'] - plot_data['CI_lower'],
                          plot_data['CI_upper'] - plot_data['Coefficient']],
                    fmt='o', capsize=5, capthick=2, markersize=8)
        
        # 添加零线
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 设置x轴标签
        plt.xticks(x_pos, plot_data['Variable'], rotation=45, ha='right')
        plt.ylabel('系数估计值', fontsize=12)
        plt.title('固定效应系数及置信区间', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加显著性标记
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            if row['Significance'] != 'ns':
                plt.text(i, row['CI_upper'] + 0.01, row['Significance'], 
                        ha='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"固定效应图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """绘制模型比较图"""
        
        if comparison_df.empty:
            print("没有模型比较数据可以绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 检查AIC值是否有效
        has_valid_aic = comparison_df['AIC'].notna().any()
        
        # AIC比较
        if has_valid_aic:
            bars1 = ax1.bar(range(len(comparison_df)), comparison_df['AIC'], color='skyblue', edgecolor='black')
            ax1.set_xlabel('模型')
            ax1.set_ylabel('AIC')
            ax1.set_title('AIC模型比较 (越小越好)')
            ax1.set_xticks(range(len(comparison_df)))
            ax1.set_xticklabels(comparison_df['Model'], rotation=45)
            
            # 标记最小值（只在有有效AIC值时）
            min_idx = comparison_df['AIC'].idxmin()
            if not pd.isna(min_idx):
                bars1[min_idx].set_color('orange')
        else:
            # 如果没有有效的AIC值，显示占位图
            ax1.bar(range(len(comparison_df)), [1]*len(comparison_df), color='gray', alpha=0.5)
            ax1.set_xlabel('模型')
            ax1.set_ylabel('AIC (不可用)')
            ax1.set_title('AIC模型比较 (数据不可用)')
            ax1.set_xticks(range(len(comparison_df)))
            ax1.set_xticklabels(comparison_df['Model'], rotation=45)
            ax1.text(0.5, 0.5, 'AIC数据不可用', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=12, alpha=0.7)
        
        # Delta AIC
        has_valid_delta = comparison_df['Delta_AIC'].notna().any()
        
        if has_valid_delta:
            bars2 = ax2.bar(range(len(comparison_df)), comparison_df['Delta_AIC'], color='lightgreen', edgecolor='black')
            ax2.set_xlabel('模型')
            ax2.set_ylabel('Delta AIC')
            ax2.set_title('Delta AIC (与最佳模型的差异)')
            ax2.set_xticks(range(len(comparison_df)))
            ax2.set_xticklabels(comparison_df['Model'], rotation=45)
            ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='实质性差异线')
            ax2.legend()
        else:
            # 如果没有有效的Delta AIC值，显示占位图
            ax2.bar(range(len(comparison_df)), [0]*len(comparison_df), color='gray', alpha=0.5)
            ax2.set_xlabel('模型')
            ax2.set_ylabel('Delta AIC (不可用)')
            ax2.set_title('Delta AIC (数据不可用)')
            ax2.set_xticks(range(len(comparison_df)))
            ax2.set_xticklabels(comparison_df['Model'], rotation=45)
            ax2.text(0.5, 0.5, 'Delta AIC数据不可用', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12, alpha=0.7)
        
        plt.suptitle('模型选择比较', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型比较图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存


if __name__ == "__main__":
    print("可视化模块已准备就绪")
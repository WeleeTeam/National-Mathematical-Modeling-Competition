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

# 设置中文字体 - 解决乱码问题
import matplotlib.font_manager as fm

# 获取系统中可用的中文字体
def get_chinese_fonts():
    """获取系统中可用的中文字体"""
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name
        # 检查是否包含中文字体
        if any(keyword in font_name.lower() for keyword in ['simhei', 'simsun', 'microsoft', 'yahei', 'kaiti', 'fangsong', 'songti']):
            chinese_fonts.append(font_name)
    return chinese_fonts

# 尝试设置中文字体
chinese_fonts = get_chinese_fonts()
if chinese_fonts:
    # 优先使用SimHei，然后是其他中文字体
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    available_fonts = [f for f in font_list if f in chinese_fonts]
    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans']
        print(f"使用中文字体: {available_fonts[0]}")
    else:
        plt.rcParams['font.sans-serif'] = chinese_fonts[:3] + ['DejaVu Sans']
        print(f"使用中文字体: {chinese_fonts[0]}")
else:
    # 如果没有找到中文字体，尝试系统默认
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
    print("警告: 未找到中文字体，使用默认设置")

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10  # 设置默认字体大小

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
        
        # 强制设置中文字体，确保每次创建实例时都应用
        self._setup_chinese_font()

    def _setup_chinese_font(self):
        """设置中文字体"""
        # 强制重新设置字体配置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
        # 清除matplotlib字体缓存
        try:
            fm._rebuild()
        except:
            pass
        
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
    
    def plot_scatter_relationships(self, data: pd.DataFrame, save_path: str = None,
                                  separate_figs: bool = False):
        """绘制散点图矩阵"""
        
        if separate_figs and save_path:
            # 如果需要分别保存，创建单独的图表
            self._plot_scatter_relationships_separate(data, save_path)
            return
        
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
    
    def plot_distribution_analysis(self, data: pd.DataFrame, save_path: str = None,
                                  separate_figs: bool = False):
        """绘制分布分析图"""
        
        if separate_figs and save_path:
            # 如果需要分别保存，创建单独的图表
            self._plot_distribution_analysis_separate(data, save_path)
            return
        
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
    
    def _plot_distribution_analysis_separate(self, data: pd.DataFrame, save_path: str):
        """绘制分布分析图（分别保存）"""
        
        # 1. Y染色体浓度分布
        plt.figure(figsize=(10, 6))
        plt.hist(data['Y染色体浓度'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.04, color='red', linestyle='--', linewidth=2, label='达标线')
        plt.xlabel('Y染色体浓度')
        plt.ylabel('频数')
        plt.title('Y染色体浓度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_01_Y浓度分布.png'), dpi=300, bbox_inches='tight')
        print(f"Y浓度分布图已保存: {save_path.replace('.png', '_01_Y浓度分布.png')}")
        plt.close()
        
        # 2. 孕周分布
        plt.figure(figsize=(10, 6))
        plt.hist(data['gestational_days'], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('孕周(天数)')
        plt.ylabel('频数')
        plt.title('孕周分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_02_孕周分布.png'), dpi=300, bbox_inches='tight')
        print(f"孕周分布图已保存: {save_path.replace('.png', '_02_孕周分布.png')}")
        plt.close()
        
        # 3. BMI分布
        plt.figure(figsize=(10, 6))
        plt.hist(data['孕妇BMI'], bins=25, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('BMI')
        plt.ylabel('频数')
        plt.title('BMI分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_03_BMI分布.png'), dpi=300, bbox_inches='tight')
        print(f"BMI分布图已保存: {save_path.replace('.png', '_03_BMI分布.png')}")
        plt.close()
        
        # 4. Y浓度的正态概率图
        plt.figure(figsize=(10, 6))
        stats.probplot(data['Y染色体浓度'], dist="norm", plot=plt)
        plt.title('Y浓度正态概率图')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_04_Y浓度正态概率图.png'), dpi=300, bbox_inches='tight')
        print(f"Y浓度正态概率图已保存: {save_path.replace('.png', '_04_Y浓度正态概率图.png')}")
        plt.close()
        
        # 5. 对数变换后的Y浓度分布
        plt.figure(figsize=(10, 6))
        log_y = np.log(data['Y染色体浓度'] + 0.001)
        plt.hist(log_y, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('log(Y染色体浓度)')
        plt.ylabel('频数')
        plt.title('对数变换Y浓度分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_05_对数变换Y浓度分布.png'), dpi=300, bbox_inches='tight')
        print(f"对数变换Y浓度分布图已保存: {save_path.replace('.png', '_05_对数变换Y浓度分布.png')}")
        plt.close()
        
        # 6. 箱线图：不同BMI组的Y浓度
        plt.figure(figsize=(10, 6))
        data['BMI_group'] = pd.cut(data['孕妇BMI'], bins=4, labels=['低', '中低', '中高', '高'])
        sns.boxplot(data=data, x='BMI_group', y='Y染色体浓度')
        plt.axhline(y=0.04, color='red', linestyle='--', alpha=0.7)
        plt.title('不同BMI组Y浓度分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_06_BMI组Y浓度箱线图.png'), dpi=300, bbox_inches='tight')
        print(f"BMI组Y浓度箱线图已保存: {save_path.replace('.png', '_06_BMI组Y浓度箱线图.png')}")
        plt.close()
        
        # 7. 13、18、21号染色体Z值箱线图
        chromosome_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']
        available_cols = [col for col in chromosome_cols if col in data.columns]
        
        if available_cols:
            plt.figure(figsize=(12, 8))
            data_melted = data[available_cols].melt(var_name='染色体', value_name='Z值')
            sns.boxplot(data=data_melted, x='染色体', y='Z值')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='正常值线')
            plt.title('13、18、21号染色体Z值分布')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', '_07_染色体Z值箱线图.png'), dpi=300, bbox_inches='tight')
            print(f"染色体Z值箱线图已保存: {save_path.replace('.png', '_07_染色体Z值箱线图.png')}")
            plt.close()
        
        # 8. X和Y染色体浓度箱线图
        xy_cols = ['X染色体浓度', 'Y染色体浓度']
        available_xy_cols = [col for col in xy_cols if col in data.columns]
        
        if available_xy_cols:
            plt.figure(figsize=(10, 6))
            data_melted = data[available_xy_cols].melt(var_name='染色体', value_name='浓度')
            sns.boxplot(data=data_melted, x='染色体', y='浓度')
            plt.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='Y染色体达标线')
            plt.title('X和Y染色体浓度分布')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', '_08_XY染色体浓度箱线图.png'), dpi=300, bbox_inches='tight')
            print(f"XY染色体浓度箱线图已保存: {save_path.replace('.png', '_08_XY染色体浓度箱线图.png')}")
            plt.close()
        
        # 9. 13、18、21号染色体Z值直方图
        if available_cols:
            fig, axes = plt.subplots(1, len(available_cols), figsize=(15, 5))
            if len(available_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(available_cols):
                axes[i].hist(data[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='正常值线')
                axes[i].set_xlabel('Z值')
                axes[i].set_ylabel('频数')
                axes[i].set_title(f'{col}分布')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', '_09_染色体Z值直方图.png'), dpi=300, bbox_inches='tight')
            print(f"染色体Z值直方图已保存: {save_path.replace('.png', '_09_染色体Z值直方图.png')}")
            plt.close()
        
        # 10. X和Y染色体浓度直方图
        if available_xy_cols:
            fig, axes = plt.subplots(1, len(available_xy_cols), figsize=(12, 5))
            if len(available_xy_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(available_xy_cols):
                axes[i].hist(data[col].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                if col == 'Y染色体浓度':
                    axes[i].axvline(x=0.04, color='red', linestyle='--', alpha=0.7, label='达标线')
                axes[i].set_xlabel('浓度')
                axes[i].set_ylabel('频数')
                axes[i].set_title(f'{col}分布')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', '_10_XY染色体浓度直方图.png'), dpi=300, bbox_inches='tight')
            print(f"XY染色体浓度直方图已保存: {save_path.replace('.png', '_10_XY染色体浓度直方图.png')}")
            plt.close()
    
    def _plot_scatter_relationships_separate(self, data: pd.DataFrame, save_path: str):
        """绘制散点图关系（分别保存）"""
        
        # 1. Y浓度 vs 孕周
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data['gestational_days'], data['Y染色体浓度'], 
                            alpha=0.6, c=data['孕妇BMI'], cmap='viridis')
        plt.colorbar(scatter, label='BMI')
        plt.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='达标线')
        plt.xlabel('孕周(天数)')
        plt.ylabel('Y染色体浓度')
        plt.title('Y浓度 vs 孕周数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_01_Y浓度vs孕周.png'), dpi=300, bbox_inches='tight')
        print(f"Y浓度vs孕周图已保存: {save_path.replace('.png', '_01_Y浓度vs孕周.png')}")
        plt.close()
        
        # 2. Y浓度 vs BMI
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data['孕妇BMI'], data['Y染色体浓度'], 
                            alpha=0.6, c=data['gestational_days'], cmap='plasma')
        plt.colorbar(scatter, label='孕周(天数)')
        plt.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='达标线')
        plt.xlabel('BMI')
        plt.ylabel('Y染色体浓度')
        plt.title('Y浓度 vs BMI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_02_Y浓度vsBMI.png'), dpi=300, bbox_inches='tight')
        print(f"Y浓度vsBMI图已保存: {save_path.replace('.png', '_02_Y浓度vsBMI.png')}")
        plt.close()
        
        # 3. Y浓度 vs 年龄
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data['年龄'], data['Y染色体浓度'], 
                            alpha=0.6, c=data['孕妇BMI'], cmap='coolwarm')
        plt.colorbar(scatter, label='BMI')
        plt.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='达标线')
        plt.xlabel('年龄')
        plt.ylabel('Y染色体浓度')
        plt.title('Y浓度 vs 年龄')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_03_Y浓度vs年龄.png'), dpi=300, bbox_inches='tight')
        print(f"Y浓度vs年龄图已保存: {save_path.replace('.png', '_03_Y浓度vs年龄.png')}")
        plt.close()
        
        # 4. BMI随孕周变化（个体轨迹）
        plt.figure(figsize=(10, 6))
        for subject in data['孕妇代码'].unique()[:20]:  # 只显示前20个个体
            subject_data = data[data['孕妇代码'] == subject]
            if len(subject_data) > 1:
                plt.plot(subject_data['gestational_days'], 
                        subject_data['孕妇BMI'], 'o-', alpha=0.5)
        plt.xlabel('孕周(天数)')
        plt.ylabel('BMI')
        plt.title('BMI随孕周变化(个体轨迹)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_04_BMI随孕周变化.png'), dpi=300, bbox_inches='tight')
        print(f"BMI随孕周变化图已保存: {save_path.replace('.png', '_04_BMI随孕周变化.png')}")
        plt.close()
    
    def plot_model_diagnostics(self, diagnostics: dict, model_name: str, save_path: str = None, separate_figs: bool = False):
        """绘制模型诊断图"""
        
        if separate_figs and save_path:
            # 如果需要分别保存，创建单独的图表
            self._plot_model_diagnostics_separate(diagnostics, model_name, save_path)
            return
        
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
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: str = None, separate_figs: bool = False):
        """绘制模型比较图"""
        
        if separate_figs and save_path:
            # 如果需要分别保存，创建单独的图表
            self._plot_model_comparison_separate(comparison_df, save_path)
            return
        
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
    
    def plot_enhanced_correlation_analysis(self, data: pd.DataFrame, 
                                         correlation_matrix: pd.DataFrame,
                                         correlation_pvalues: pd.DataFrame,
                                         bmi_group_correlations: dict,
                                         save_path: str = None,
                                         separate_figs: bool = False):
        """绘制增强版相关性分析图表"""
        
        if separate_figs and save_path:
            # 如果需要分别保存，创建单独的图表
            self._plot_enhanced_correlation_analysis_separate(
                data, correlation_matrix, correlation_pvalues, 
                bmi_group_correlations, save_path
            )
            return
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 整体相关性热力图 (左上)
        plt.subplot(2, 3, 1)
        # 选择与Y染色体浓度相关的主要变量
        main_vars = ['Y染色体浓度', 'gestational_days', '孕妇BMI', '年龄', '身高', '体重']
        if 'Y染色体的Z值' in correlation_matrix.index:
            main_vars.append('Y染色体的Z值')
        
        available_vars = [v for v in main_vars if v in correlation_matrix.index]
        main_corr = correlation_matrix.loc[available_vars, available_vars]
        mask = np.triu(np.ones_like(main_corr, dtype=bool))
        sns.heatmap(main_corr, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('主要变量相关性矩阵', fontsize=12, fontweight='bold')
        
        # 2. Y染色体浓度相关性条形图 (右上)
        plt.subplot(2, 3, 2)
        y_correlations = correlation_matrix['Y染色体浓度'].drop('Y染色体浓度').dropna()
        
        if 'Y染色体浓度' in correlation_pvalues.index:
            y_pvalues = correlation_pvalues['Y染色体浓度'].loc[y_correlations.index]
        else:
            y_pvalues = pd.Series([np.nan]*len(y_correlations), index=y_correlations.index)
        
        # 根据显著性设置颜色
        colors = []
        for p in y_pvalues:
            if pd.isna(p):
                colors.append('gray')
            elif p < 0.001:
                colors.append('red')
            elif p < 0.01:
                colors.append('orange')
            elif p < 0.05:
                colors.append('yellow')
            else:
                colors.append('lightgray')
        
        bars = plt.barh(range(len(y_correlations)), y_correlations.values, color=colors)
        plt.yticks(range(len(y_correlations)), y_correlations.index, rotation=0)
        plt.xlabel('相关系数 r')
        plt.title('Y染色体浓度相关性（按显著性着色）', fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # 3. BMI分组相关性比较 (左下)
        plt.subplot(2, 3, 4)
        if bmi_group_correlations:
            groups = list(bmi_group_correlations.keys())
            gest_corrs = [bmi_group_correlations[g].get('gestational_days', 0) for g in groups]
            
            plt.bar(groups, gest_corrs, alpha=0.7, color='skyblue')
            plt.xlabel('BMI组别')
            plt.ylabel('Y浓度-孕周相关性 r')
            plt.title('不同BMI组的相关性差异', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 4. Y染色体浓度 vs 孕周散点图 (中下)
        plt.subplot(2, 3, 5)
        scatter = plt.scatter(data['gestational_days'], data['Y染色体浓度'], 
                            c=data['孕妇BMI'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='BMI')
        plt.axhline(y=0.04, color='red', linestyle='--', linewidth=2, label='达标阈值(4%)')
        plt.xlabel('孕周(天数)')
        plt.ylabel('Y染色体浓度')
        plt.title('Y浓度 vs 孕周数（按BMI着色）', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 非线性趋势分析 (右下)
        plt.subplot(2, 3, 6)
        # 拟合多项式回归线
        x = data['gestational_days'].dropna()
        y_all = data['Y染色体浓度']
        
        # 找到两个变量都非缺失的索引
        common_idx = x.index.intersection(y_all.dropna().index)
        x_clean = x.loc[common_idx]
        y_clean = y_all.loc[common_idx]
        
        if len(x_clean) > 20:
            # 线性拟合
            z_linear = np.polyfit(x_clean, y_clean, 1)
            p_linear = np.poly1d(z_linear)
            
            # 二次拟合
            try:
                z_quad = np.polyfit(x_clean, y_clean, 2)  
                p_quad = np.poly1d(z_quad)
                has_quad = True
            except:
                has_quad = False
            
            x_smooth = np.linspace(x_clean.min(), x_clean.max(), 100)
            
            plt.scatter(x_clean, y_clean, alpha=0.5, color='lightblue', s=20)
            plt.plot(x_smooth, p_linear(x_smooth), 'r--', linewidth=2, label='线性拟合')
            
            if has_quad:
                plt.plot(x_smooth, p_quad(x_smooth), 'g-', linewidth=2, label='二次拟合')
            
            plt.axhline(y=0.04, color='red', linestyle=':', alpha=0.7, label='达标线')
            
            plt.xlabel('孕周(天数)')
            plt.ylabel('Y染色体浓度')
            plt.title('线性 vs 非线性关系', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. 显著性水平图例 (中上)
        plt.subplot(2, 3, 3)
        plt.axis('off')
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='red', label='p < 0.001 (***)'),
            plt.Rectangle((0, 0), 1, 1, color='orange', label='p < 0.01 (**)'),
            plt.Rectangle((0, 0), 1, 1, color='yellow', label='p < 0.05 (*)'),
            plt.Rectangle((0, 0), 1, 1, color='lightgray', label='p ≥ 0.05 (ns)')
        ]
        plt.legend(handles=legend_elements, loc='center', fontsize=14)
        plt.title('显著性水平说明', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"增强相关性分析图已保存: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def _plot_enhanced_correlation_analysis_separate(self, data: pd.DataFrame, 
                                                   correlation_matrix: pd.DataFrame,
                                                   correlation_pvalues: pd.DataFrame,
                                                   bmi_group_correlations: dict,
                                                   save_path: str):
        """绘制增强版相关性分析图表（分别保存）"""
        
        # 1. 主要变量相关性矩阵
        plt.figure(figsize=(12, 10))
        main_vars = ['Y染色体浓度', 'gestational_days', '孕妇BMI', '年龄', '身高', '体重']
        if 'Y染色体的Z值' in correlation_matrix.index:
            main_vars.append('Y染色体的Z值')
        
        available_vars = [v for v in main_vars if v in correlation_matrix.index]
        main_corr = correlation_matrix.loc[available_vars, available_vars]
        mask = np.triu(np.ones_like(main_corr, dtype=bool))
        sns.heatmap(main_corr, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('主要变量相关性矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_01_主要变量相关性矩阵.png'), dpi=300, bbox_inches='tight')
        print(f"主要变量相关性矩阵已保存: {save_path.replace('.png', '_01_主要变量相关性矩阵.png')}")
        plt.close()
        
        # 2. Y染色体浓度相关性条形图
        plt.figure(figsize=(12, 8))
        y_correlations = correlation_matrix['Y染色体浓度'].drop('Y染色体浓度').dropna()
        
        if 'Y染色体浓度' in correlation_pvalues.index:
            y_pvalues = correlation_pvalues['Y染色体浓度'].loc[y_correlations.index]
        else:
            y_pvalues = pd.Series([np.nan]*len(y_correlations), index=y_correlations.index)
        
        # 根据显著性设置颜色
        colors = []
        for p in y_pvalues:
            if pd.isna(p):
                colors.append('gray')
            elif p < 0.001:
                colors.append('red')
            elif p < 0.01:
                colors.append('orange')
            elif p < 0.05:
                colors.append('yellow')
            else:
                colors.append('lightgray')
        
        bars = plt.barh(range(len(y_correlations)), y_correlations.values, color=colors)
        plt.yticks(range(len(y_correlations)), y_correlations.index, rotation=0)
        plt.xlabel('相关系数 r')
        plt.title('Y染色体浓度相关性（按显著性着色）', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_02_Y浓度相关性条形图.png'), dpi=300, bbox_inches='tight')
        print(f"Y浓度相关性条形图已保存: {save_path.replace('.png', '_02_Y浓度相关性条形图.png')}")
        plt.close()
        
        # 3. 显著性水平说明
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='red', label='p < 0.001 (***)'),
            plt.Rectangle((0, 0), 1, 1, color='orange', label='p < 0.01 (**)'),
            plt.Rectangle((0, 0), 1, 1, color='yellow', label='p < 0.05 (*)'),
            plt.Rectangle((0, 0), 1, 1, color='lightgray', label='p ≥ 0.05 (ns)')
        ]
        plt.legend(handles=legend_elements, loc='center', fontsize=14)
        plt.title('显著性水平说明', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_03_显著性水平说明.png'), dpi=300, bbox_inches='tight')
        print(f"显著性水平说明已保存: {save_path.replace('.png', '_03_显著性水平说明.png')}")
        plt.close()
        
        # 4. BMI分组相关性比较
        if bmi_group_correlations:
            plt.figure(figsize=(10, 6))
            groups = list(bmi_group_correlations.keys())
            gest_corrs = [bmi_group_correlations[g].get('gestational_days', 0) for g in groups]
            
            plt.bar(groups, gest_corrs, alpha=0.7, color='skyblue')
            plt.xlabel('BMI组别')
            plt.ylabel('Y浓度-孕周相关性 r')
            plt.title('不同BMI组的相关性差异', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', '_04_BMI分组相关性比较.png'), dpi=300, bbox_inches='tight')
            print(f"BMI分组相关性比较已保存: {save_path.replace('.png', '_04_BMI分组相关性比较.png')}")
            plt.close()
        
        # 5. Y染色体浓度 vs 孕周散点图
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(data['gestational_days'], data['Y染色体浓度'], 
                            c=data['孕妇BMI'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='BMI')
        plt.axhline(y=0.04, color='red', linestyle='--', linewidth=2, label='达标阈值(4%)')
        plt.xlabel('孕周(天数)')
        plt.ylabel('Y染色体浓度')
        plt.title('Y浓度 vs 孕周数（按BMI着色）', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_05_Y浓度vs孕周散点图.png'), dpi=300, bbox_inches='tight')
        print(f"Y浓度vs孕周散点图已保存: {save_path.replace('.png', '_05_Y浓度vs孕周散点图.png')}")
        plt.close()
        
        # 6. 非线性趋势分析
        plt.figure(figsize=(12, 8))
        # 拟合多项式回归线
        x = data['gestational_days'].dropna()
        y_all = data['Y染色体浓度']
        
        # 找到两个变量都非缺失的索引
        common_idx = x.index.intersection(y_all.dropna().index)
        x_clean = x.loc[common_idx]
        y_clean = y_all.loc[common_idx]
        
        if len(x_clean) > 20:
            # 线性拟合
            z_linear = np.polyfit(x_clean, y_clean, 1)
            p_linear = np.poly1d(z_linear)
            
            # 二次拟合
            try:
                z_quad = np.polyfit(x_clean, y_clean, 2)  
                p_quad = np.poly1d(z_quad)
                has_quad = True
            except:
                has_quad = False
            
            x_smooth = np.linspace(x_clean.min(), x_clean.max(), 100)
            
            plt.scatter(x_clean, y_clean, alpha=0.5, color='lightblue', s=20)
            plt.plot(x_smooth, p_linear(x_smooth), 'r--', linewidth=2, label='线性拟合')
            
            if has_quad:
                plt.plot(x_smooth, p_quad(x_smooth), 'g-', linewidth=2, label='二次拟合')
            
            plt.axhline(y=0.04, color='red', linestyle=':', alpha=0.7, label='达标线')
            
            plt.xlabel('孕周(天数)')
            plt.ylabel('Y染色体浓度')
            plt.title('线性 vs 非线性关系', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', '_06_非线性趋势分析.png'), dpi=300, bbox_inches='tight')
            print(f"非线性趋势分析已保存: {save_path.replace('.png', '_06_非线性趋势分析.png')}")
            plt.close()
    
    def _plot_model_diagnostics_separate(self, diagnostics: dict, model_name: str, save_path: str):
        """绘制模型诊断图（分别保存）"""
        
        residuals = diagnostics['residuals']
        fitted_values = diagnostics['fitted_values']
        std_residuals = diagnostics['std_residuals']
        
        # 1. 残差 vs 拟合值
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('拟合值')
        plt.ylabel('残差')
        plt.title('残差 vs 拟合值')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_01_残差vs拟合值.png'), dpi=300, bbox_inches='tight')
        print(f"残差vs拟合值图已保存: {save_path.replace('.png', '_01_残差vs拟合值.png')}")
        plt.close()
        
        # 2. 标准化残差 vs 拟合值
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, std_residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.axhline(y=2, color='red', linestyle=':', alpha=0.5)
        plt.axhline(y=-2, color='red', linestyle=':', alpha=0.5)
        plt.xlabel('拟合值')
        plt.ylabel('标准化残差')
        plt.title('标准化残差 vs 拟合值')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_02_标准化残差vs拟合值.png'), dpi=300, bbox_inches='tight')
        print(f"标准化残差vs拟合值图已保存: {save_path.replace('.png', '_02_标准化残差vs拟合值.png')}")
        plt.close()
        
        # 3. 残差正态概率图
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('残差正态概率图')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_03_残差正态概率图.png'), dpi=300, bbox_inches='tight')
        print(f"残差正态概率图已保存: {save_path.replace('.png', '_03_残差正态概率图.png')}")
        plt.close()
        
        # 4. 残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=25, alpha=0.7, color='lightblue', edgecolor='black')
        plt.xlabel('残差')
        plt.ylabel('频数')
        plt.title('残差分布')
        
        # 添加正态性检验结果
        if 'shapiro_p_value' in diagnostics and not np.isnan(diagnostics['shapiro_p_value']):
            plt.text(0.05, 0.95, f"Shapiro-Wilk p={diagnostics['shapiro_p_value']:.4f}", 
                    transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_04_残差分布.png'), dpi=300, bbox_inches='tight')
        print(f"残差分布图已保存: {save_path.replace('.png', '_04_残差分布.png')}")
        plt.close()
    
    def _plot_model_comparison_separate(self, comparison_df: pd.DataFrame, save_path: str):
        """绘制模型比较图（分别保存）"""
        
        if comparison_df.empty:
            print("没有模型比较数据可以绘制")
            return
        
        # 1. AIC比较
        plt.figure(figsize=(10, 6))
        has_valid_aic = comparison_df['AIC'].notna().any()
        
        if has_valid_aic:
            bars = plt.bar(range(len(comparison_df)), comparison_df['AIC'], color='skyblue', edgecolor='black')
            plt.xlabel('模型')
            plt.ylabel('AIC')
            plt.title('AIC模型比较 (越小越好)')
            plt.xticks(range(len(comparison_df)), comparison_df['Model'], rotation=45)
            
            # 标记最小值
            min_idx = comparison_df['AIC'].idxmin()
            if not pd.isna(min_idx):
                bars[min_idx].set_color('orange')
        else:
            plt.bar(range(len(comparison_df)), [1]*len(comparison_df), color='gray', alpha=0.5)
            plt.xlabel('模型')
            plt.ylabel('AIC (不可用)')
            plt.title('AIC模型比较 (数据不可用)')
            plt.xticks(range(len(comparison_df)), comparison_df['Model'], rotation=45)
            plt.text(0.5, 0.5, 'AIC数据不可用', transform=plt.gca().transAxes, 
                    ha='center', va='center', fontsize=12, alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_01_AIC比较.png'), dpi=300, bbox_inches='tight')
        print(f"AIC比较图已保存: {save_path.replace('.png', '_01_AIC比较.png')}")
        plt.close()
        
        # 2. Delta AIC
        plt.figure(figsize=(10, 6))
        has_valid_delta = comparison_df['Delta_AIC'].notna().any()
        
        if has_valid_delta:
            bars = plt.bar(range(len(comparison_df)), comparison_df['Delta_AIC'], color='lightgreen', edgecolor='black')
            plt.xlabel('模型')
            plt.ylabel('Delta AIC')
            plt.title('Delta AIC (与最佳模型的差异)')
            plt.xticks(range(len(comparison_df)), comparison_df['Model'], rotation=45)
            plt.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='实质性差异线')
            plt.legend()
        else:
            plt.bar(range(len(comparison_df)), [0]*len(comparison_df), color='gray', alpha=0.5)
            plt.xlabel('模型')
            plt.ylabel('Delta AIC (不可用)')
            plt.title('Delta AIC (数据不可用)')
            plt.xticks(range(len(comparison_df)), comparison_df['Model'], rotation=45)
            plt.text(0.5, 0.5, 'Delta AIC数据不可用', transform=plt.gca().transAxes, 
                    ha='center', va='center', fontsize=12, alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_02_DeltaAIC.png'), dpi=300, bbox_inches='tight')
        print(f"DeltaAIC图已保存: {save_path.replace('.png', '_02_DeltaAIC.png')}")
        plt.close()


if __name__ == "__main__":
    print("可视化模块已准备就绪")
"""
问题3可视化模块
多因素分析结果的可视化展示
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Problem3Visualizer:
    """问题3多因素分析可视化器"""
    
    def __init__(self, figsize_default: tuple = (12, 8), dpi: int = 300):
        """
        初始化可视化器
        
        Parameters:
        - figsize_default: 默认图形大小
        - dpi: 图形分辨率
        """
        self.figsize_default = figsize_default
        self.dpi = dpi
        
        # 设置颜色主题
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        print("问题3多因素可视化器初始化完成")
    
    def plot_multifactor_达标时间_vs_bmi(self, prediction_results: pd.DataFrame, 
                                        save_path: str = None) -> plt.Figure:
        """
        绘制多因素达标时间与BMI的关系图
        
        Parameters:
        - prediction_results: 预测结果数据
        - save_path: 保存路径
        
        Returns:
        - fig: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('多因素达标时间分析', fontsize=16, fontweight='bold')
        
        # 1. 达标时间 vs BMI
        ax1 = axes[0, 0]
        valid_data = prediction_results.dropna(subset=['预测达标周数'])
        scatter = ax1.scatter(valid_data['BMI'], valid_data['预测达标周数'], 
                             c=valid_data['年龄'], cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('BMI')
        ax1.set_ylabel('预测达标周数')
        ax1.set_title('达标时间 vs BMI (颜色表示年龄)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='年龄')
        
        # 2. 达标时间 vs 年龄
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(valid_data['年龄'], valid_data['预测达标周数'], 
                              c=valid_data['BMI'], cmap='plasma', alpha=0.7, s=50)
        ax2.set_xlabel('年龄')
        ax2.set_ylabel('预测达标周数')
        ax2.set_title('达标时间 vs 年龄 (颜色表示BMI)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='BMI')
        
        # 3. 达标时间 vs 身高
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(valid_data['身高'], valid_data['预测达标周数'], 
                              c=valid_data['体重'], cmap='coolwarm', alpha=0.7, s=50)
        ax3.set_xlabel('身高 (cm)')
        ax3.set_ylabel('预测达标周数')
        ax3.set_title('达标时间 vs 身高 (颜色表示体重)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='体重 (kg)')
        
        # 4. 达标时间 vs 体重
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(valid_data['体重'], valid_data['预测达标周数'], 
                              c=valid_data['BMI'], cmap='inferno', alpha=0.7, s=50)
        ax4.set_xlabel('体重 (kg)')
        ax4.set_ylabel('预测达标周数')
        ax4.set_title('达标时间 vs 体重 (颜色表示BMI)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='BMI')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_multifactor_grouping_results(self, grouping_rules: List[Dict], 
                                        refined_groups: Dict, save_path: str = None) -> plt.Figure:
        """
        绘制多因素分组结果
        
        Parameters:
        - grouping_rules: 分组规则
        - refined_groups: 细化分组结果
        - save_path: 保存路径
        
        Returns:
        - fig: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('多因素分组结果分析', fontsize=16, fontweight='bold')
        
        # 1. BMI分组分布
        ax1 = axes[0, 0]
        group_names = [rule['group_name'] for rule in grouping_rules]
        bmi_means = [rule['bmi_mean'] for rule in grouping_rules]
        sample_sizes = [rule['sample_size'] for rule in grouping_rules]
        
        bars = ax1.bar(group_names, sample_sizes, color=self.colors[:len(group_names)])
        ax1.set_xlabel('分组')
        ax1.set_ylabel('样本数')
        ax1.set_title('各分组样本数分布')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱子上添加BMI均值标签
        for i, (bar, bmi_mean) in enumerate(zip(bars, bmi_means)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'BMI: {bmi_mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 推荐检测时间分布
        ax2 = axes[0, 1]
        test_times = [rule['recommended_test_time_weeks'] for rule in grouping_rules]
        bars2 = ax2.bar(group_names, test_times, color=self.colors[:len(group_names)])
        ax2.set_xlabel('分组')
        ax2.set_ylabel('推荐检测时间 (周)')
        ax2.set_title('各分组推荐检测时间')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 多因素评分分布
        ax3 = axes[1, 0]
        multifactor_scores = [rule['multifactor_score_mean'] for rule in grouping_rules]
        bars3 = ax3.bar(group_names, multifactor_scores, color=self.colors[:len(group_names)])
        ax3.set_xlabel('分组')
        ax3.set_ylabel('多因素综合评分')
        ax3.set_title('各分组多因素综合评分')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 期望风险分布
        ax4 = axes[1, 1]
        expected_risks = [rule['expected_risk'] for rule in grouping_rules]
        bars4 = ax4.bar(group_names, expected_risks, color=self.colors[:len(group_names)])
        ax4.set_xlabel('分组')
        ax4.set_ylabel('期望风险')
        ax4.set_title('各分组期望风险')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_multifactor_correlation_heatmap(self, feature_df: pd.DataFrame, 
                                           save_path: str = None) -> plt.Figure:
        """
        绘制多因素相关性热力图
        
        Parameters:
        - feature_df: 特征数据框
        - save_path: 保存路径
        
        Returns:
        - fig: 图形对象
        """
        # 选择数值特征
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns
        correlation_data = feature_df[numeric_features].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 创建热力图
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        ax.set_title('多因素特征相关性热力图', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_multifactor_sensitivity_analysis(self, sensitivity_results: Dict, 
                                            save_path: str = None) -> plt.Figure:
        """
        绘制多因素敏感性分析图
        
        Parameters:
        - sensitivity_results: 敏感性分析结果
        - save_path: 保存路径
        
        Returns:
        - fig: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('多因素检测误差敏感性分析', fontsize=16, fontweight='bold')
        
        # 获取第一个组的数据作为示例
        if sensitivity_results:
            first_group = list(sensitivity_results.keys())[0]
            sens_data = sensitivity_results[first_group]
            
            # 1. 风险 vs 测量误差
            ax1 = axes[0, 0]
            ax1.plot(sens_data['measurement_error_std'], sens_data['expected_risk'], 
                    'o-', linewidth=2, markersize=6, color=self.colors[0])
            ax1.fill_between(sens_data['measurement_error_std'], 
                            sens_data['expected_risk'] - sens_data['risk_std'],
                            sens_data['expected_risk'] + sens_data['risk_std'],
                            alpha=0.3, color=self.colors[0])
            ax1.set_xlabel('测量误差标准差')
            ax1.set_ylabel('期望风险')
            ax1.set_title(f'{first_group} 风险敏感性')
            ax1.grid(True, alpha=0.3)
            
            # 2. 成功率 vs 测量误差
            ax2 = axes[0, 1]
            ax2.plot(sens_data['measurement_error_std'], sens_data['success_rate'], 
                    'o-', linewidth=2, markersize=6, color=self.colors[1])
            ax2.fill_between(sens_data['measurement_error_std'], 
                            sens_data['success_rate'] - sens_data['success_rate_std'],
                            sens_data['success_rate'] + sens_data['success_rate_std'],
                            alpha=0.3, color=self.colors[1])
            ax2.set_xlabel('测量误差标准差')
            ax2.set_ylabel('成功率')
            ax2.set_title(f'{first_group} 成功率敏感性')
            ax2.grid(True, alpha=0.3)
            
            # 3. 风险变化率
            ax3 = axes[1, 0]
            risk_change = np.diff(sens_data['expected_risk']) / np.diff(sens_data['measurement_error_std'])
            ax3.plot(sens_data['measurement_error_std'][1:], risk_change, 
                    'o-', linewidth=2, markersize=6, color=self.colors[2])
            ax3.set_xlabel('测量误差标准差')
            ax3.set_ylabel('风险变化率')
            ax3.set_title(f'{first_group} 风险变化率')
            ax3.grid(True, alpha=0.3)
            
            # 4. 综合敏感性指标
            ax4 = axes[1, 1]
            sensitivity_index = sens_data['expected_risk'] / sens_data['success_rate']
            ax4.plot(sens_data['measurement_error_std'], sensitivity_index, 
                    'o-', linewidth=2, markersize=6, color=self.colors[3])
            ax4.set_xlabel('测量误差标准差')
            ax4.set_ylabel('敏感性指数 (风险/成功率)')
            ax4.set_title(f'{first_group} 综合敏感性')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def create_multifactor_comprehensive_dashboard(self, analysis_results: Dict, 
                                                 save_path: str = None) -> plt.Figure:
        """
        创建多因素综合分析仪表板
        
        Parameters:
        - analysis_results: 分析结果
        - save_path: 保存路径
        
        Returns:
        - fig: 图形对象
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('问题3多因素分析综合仪表板', fontsize=20, fontweight='bold')
        
        # 1. 数据概览
        ax1 = fig.add_subplot(gs[0, 0])
        valid_predictions = analysis_results.get('valid_predictions', pd.DataFrame())
        total_patients = len(valid_predictions)
        ax1.text(0.5, 0.7, f'总患者数\n{total_patients}', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.3, f'有效预测\n{len(valid_predictions)}', ha='center', va='center', 
                fontsize=12, transform=ax1.transAxes)
        ax1.set_title('数据概览', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. 分组数量
        ax2 = fig.add_subplot(gs[0, 1])
        grouping_rules = analysis_results.get('grouping_rules', [])
        n_groups = len(grouping_rules)
        ax2.text(0.5, 0.5, f'分组数量\n{n_groups}', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=ax2.transAxes)
        ax2.set_title('分组统计', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. 约束满足情况
        ax3 = fig.add_subplot(gs[0, 2])
        constraint_validation = analysis_results.get('constraint_validation', {})
        satisfied_constraints = sum([
            constraint_validation.get('bmi_segmentation_valid', False),
            constraint_validation.get('detection_time_valid', False),
            constraint_validation.get('multifactor_robust_constraint_valid', False)
        ])
        ax3.text(0.5, 0.5, f'约束满足\n{satisfied_constraints}/3', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=ax3.transAxes)
        ax3.set_title('约束验证', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. 平均风险
        ax4 = fig.add_subplot(gs[0, 3])
        optimization_results = analysis_results.get('optimization_results', {})
        if optimization_results:
            avg_risk = np.mean([opt['minimal_expected_risk'] for opt in optimization_results.values()])
            ax4.text(0.5, 0.5, f'平均风险\n{avg_risk:.4f}', ha='center', va='center', 
                    fontsize=14, fontweight='bold', transform=ax4.transAxes)
        ax4.set_title('风险统计', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. BMI分布直方图
        ax5 = fig.add_subplot(gs[1, :2])
        if not valid_predictions.empty:
            ax5.hist(valid_predictions['BMI'], bins=20, alpha=0.7, color=self.colors[0])
            ax5.set_xlabel('BMI')
            ax5.set_ylabel('频数')
            ax5.set_title('BMI分布')
            ax5.grid(True, alpha=0.3)
        
        # 6. 达标时间分布
        ax6 = fig.add_subplot(gs[1, 2:])
        if not valid_predictions.empty and '预测达标周数' in valid_predictions.columns:
            valid_times = valid_predictions['预测达标周数'].dropna()
            ax6.hist(valid_times, bins=20, alpha=0.7, color=self.colors[1])
            ax6.set_xlabel('达标时间 (周)')
            ax6.set_ylabel('频数')
            ax6.set_title('达标时间分布')
            ax6.grid(True, alpha=0.3)
        
        # 7. 分组结果表格
        ax7 = fig.add_subplot(gs[2:, :])
        if grouping_rules:
            # 创建分组结果表格
            table_data = []
            for rule in grouping_rules:
                table_data.append([
                    rule['group_name'],
                    rule['bmi_interval_description'],
                    f"{rule['sample_size']}",
                    f"{rule['recommended_test_time_weeks']:.1f}周",
                    f"{rule['expected_risk']:.4f}",
                    f"{rule['multifactor_score_mean']:.3f}"
                ])
            
            table = ax7.table(cellText=table_data,
                            colLabels=['分组', 'BMI范围', '样本数', '推荐时间', '期望风险', '多因素评分'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # 设置表格样式
            for i in range(len(table_data) + 1):
                for j in range(6):
                    cell = table[(i, j)]
                    if i == 0:  # 表头
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            ax7.set_title('多因素分组结果详情', fontsize=14, fontweight='bold', pad=20)
            ax7.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # 测试问题3可视化
    print("问题3可视化模块测试")
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'BMI': np.random.normal(32, 5, 100),
        '年龄': np.random.normal(30, 4, 100),
        '身高': np.random.normal(165, 8, 100),
        '体重': np.random.normal(70, 10, 100),
        '预测达标周数': np.random.normal(15, 3, 100)
    })
    
    visualizer = Problem3Visualizer()
    
    # 测试达标时间分析图
    fig = visualizer.plot_multifactor_达标时间_vs_bmi(test_data)
    print("多因素达标时间分析图生成完成")
    
    plt.show()
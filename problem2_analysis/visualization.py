"""
可视化模块
生成问题2的各种分析图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
sns.set_palette("husl")


class Problem2Visualizer:
    """问题2可视化器"""
    
    def __init__(self, figsize_default: tuple = (12, 8), dpi: int = 300):
        self.figsize_default = figsize_default
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 8)
        
    def plot_达标时间_vs_bmi(self, prediction_results: pd.DataFrame, 
                           save_path: str = None) -> plt.Figure:
        """绘制预测达标时间与BMI的关系（包含95%约束时间）"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 过滤有效数据
        valid_data = prediction_results.dropna(subset=['满足95%约束时间'])
        
        # 图1：95%约束时间 vs BMI散点图
        ax1.scatter(valid_data['孕妇BMI'], valid_data['满足95%约束周数'], 
                   alpha=0.6, s=50, c=self.color_palette[0], label='95%约束时间')
        
        # 添加趋势线
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['孕妇BMI'], valid_data['满足95%约束周数'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(valid_data['孕妇BMI'].min(), valid_data['孕妇BMI'].max(), 100)
            ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                    label=f'趋势线: y={z[0]:.3f}x+{z[1]:.1f}')
        
        ax1.set_xlabel('孕妇BMI', fontsize=12, fontweight='bold')
        ax1.set_ylabel('满足95%约束时间 (周)', fontsize=12, fontweight='bold')
        ax1.set_title('满足95%成功概率的检测时间 vs BMI', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 图2：BMI分组的95%约束时间箱线图
        bmi_bins = pd.cut(valid_data['孕妇BMI'], bins=5, precision=1)
        box_data = []
        box_labels = []
        
        for bin_name in bmi_bins.cat.categories:
            bin_data = valid_data[bmi_bins == bin_name]['满足95%约束周数']
            if len(bin_data) > 0:
                box_data.append(bin_data)
                box_labels.append(f'{bin_name}')
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # 设置箱线图颜色
            for patch, color in zip(bp['boxes'], self.color_palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_xlabel('BMI区间', fontsize=12, fontweight='bold')
        ax2.set_ylabel('满足95%约束时间 (周)', fontsize=12, fontweight='bold') 
        ax2.set_title('不同BMI区间的95%约束时间分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 图3：对比传统达标时间和95%约束时间
        if '预测达标周数' in valid_data.columns:
            traditional_data = valid_data.dropna(subset=['预测达标周数'])
            if len(traditional_data) > 0:
                ax3.scatter(traditional_data['孕妇BMI'], traditional_data['预测达标周数'], 
                           alpha=0.5, s=30, c=self.color_palette[1], label='传统达标时间')
                ax3.scatter(traditional_data['孕妇BMI'], traditional_data['满足95%约束周数'], 
                           alpha=0.7, s=30, c=self.color_palette[0], label='95%约束时间')
                ax3.legend()
        
        ax3.set_xlabel('孕妇BMI', fontsize=12, fontweight='bold')
        ax3.set_ylabel('检测时间 (周)', fontsize=12, fontweight='bold')
        ax3.set_title('传统达标时间 vs 95%约束时间', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 图4：BMI分布直方图
        ax4.hist(valid_data['孕妇BMI'], bins=20, alpha=0.7, color=self.color_palette[2], edgecolor='black')
        ax4.set_xlabel('孕妇BMI', fontsize=12, fontweight='bold')
        ax4.set_ylabel('患者数量', fontsize=12, fontweight='bold')
        ax4.set_title('BMI分布', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_risk_vs_test_time(self, risk_analysis: Dict, save_path: str = None) -> plt.Figure:
        """绘制风险随检测时间的变化"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取分析结果数据
        test_times = []
        expected_risks = []
        success_rates = []
        error_risks = []
        
        for group_name, group_analysis in risk_analysis.items():
            if 'test_time_analysis' in group_analysis:
                time_data = group_analysis['test_time_analysis']
                test_times.extend(time_data['test_times'])
                expected_risks.extend(time_data['expected_risks'])
                success_rates.extend(time_data['success_rates'])
                error_risks.extend(time_data['error_risks'])
        
        if test_times:  # 如果有数据
            # 图1: 期望风险 vs 检测时间
            ax1.plot(np.array(test_times)/7, expected_risks, 'o-', 
                    linewidth=2, markersize=6, color=self.color_palette[0])
            ax1.set_xlabel('检测时间 (周)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('期望风险', fontsize=12, fontweight='bold')
            ax1.set_title('期望风险 vs 检测时间', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 图2: 成功率 vs 检测时间
            ax2.plot(np.array(test_times)/7, success_rates, 'o-', 
                    linewidth=2, markersize=6, color=self.color_palette[1])
            ax2.set_xlabel('检测时间 (周)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('达标成功率', fontsize=12, fontweight='bold')
            ax2.set_title('达标成功率 vs 检测时间', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
        
        # 图3和图4：展示不同BMI组的风险对比
        if len(risk_analysis) > 1:
            group_names = list(risk_analysis.keys())
            group_risks = [risk_analysis[name].get('minimal_risk', 0) for name in group_names]
            group_test_times = [risk_analysis[name].get('optimal_test_time_weeks', 15) for name in group_names]
            
            ax3.bar(range(len(group_names)), group_risks, 
                   color=self.color_palette[:len(group_names)], alpha=0.7)
            ax3.set_xlabel('BMI组别', fontsize=12, fontweight='bold')
            ax3.set_ylabel('最小期望风险', fontsize=12, fontweight='bold')
            ax3.set_title('各BMI组最小期望风险', fontsize=14, fontweight='bold')
            ax3.set_xticks(range(len(group_names)))
            ax3.set_xticklabels(group_names, rotation=45)
            ax3.grid(True, alpha=0.3)
            
            ax4.bar(range(len(group_names)), group_test_times,
                   color=self.color_palette[:len(group_names)], alpha=0.7)
            ax4.set_xlabel('BMI组别', fontsize=12, fontweight='bold')
            ax4.set_ylabel('最优检测时间 (周)', fontsize=12, fontweight='bold')
            ax4.set_title('各BMI组最优检测时间', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(len(group_names)))
            ax4.set_xticklabels(group_names, rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_grouping_results(self, grouping_rules: List[Dict], 
                             refined_groups: Dict, save_path: str = None) -> plt.Figure:
        """可视化分组结果"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 图1: BMI组分布
        group_names = [rule['group_name'] for rule in grouping_rules]
        sample_sizes = [rule['sample_size'] for rule in grouping_rules]
        
        colors = self.color_palette[:len(group_names)]
        wedges, texts, autotexts = ax1.pie(sample_sizes, labels=group_names, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax1.set_title('BMI组样本分布', fontsize=14, fontweight='bold')
        
        # 图2: 各组BMI范围
        bmi_ranges = [(rule['bmi_lower_bound'], rule['bmi_upper_bound']) for rule in grouping_rules]
        bmi_means = [(r[0] + r[1])/2 for r in bmi_ranges]
        bmi_widths = [r[1] - r[0] for r in bmi_ranges]
        
        bars = ax2.bar(range(len(group_names)), bmi_widths, bottom=[r[0] for r in bmi_ranges],
                      color=colors, alpha=0.7, edgecolor='black')
        
        # 添加BMI范围标注
        for i, (rule, bar) in enumerate(zip(grouping_rules, bars)):
            height = bar.get_height()
            bottom = bar.get_y()
            ax2.text(bar.get_x() + bar.get_width()/2., bottom + height/2.,
                    f'{rule["bmi_interval_description"]}', 
                    ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('BMI组别', fontsize=12, fontweight='bold')
        ax2.set_ylabel('BMI值', fontsize=12, fontweight='bold')
        ax2.set_title('各BMI组的BMI范围', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(group_names)))
        ax2.set_xticklabels(group_names)
        ax2.grid(True, alpha=0.3)
        
        # 图3: 推荐检测时间
        test_times = [rule['recommended_test_time_weeks'] for rule in grouping_rules]
        bars3 = ax3.bar(range(len(group_names)), test_times, color=colors, alpha=0.7)
        
        # 添加数值标注
        for bar, time in zip(bars3, test_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}周', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_xlabel('BMI组别', fontsize=12, fontweight='bold')
        ax3.set_ylabel('推荐检测时间 (周)', fontsize=12, fontweight='bold')
        ax3.set_title('各BMI组推荐检测时间', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(group_names)))
        ax3.set_xticklabels(group_names)
        ax3.grid(True, alpha=0.3)
        
        # 图4: 期望风险对比
        expected_risks = [rule['expected_risk'] for rule in grouping_rules]
        bars4 = ax4.bar(range(len(group_names)), expected_risks, color=colors, alpha=0.7)
        
        # 添加数值标注
        for bar, risk in zip(bars4, expected_risks):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{risk:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_xlabel('BMI组别', fontsize=12, fontweight='bold')
        ax4.set_ylabel('期望风险', fontsize=12, fontweight='bold')
        ax4.set_title('各BMI组期望风险', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(group_names)))
        ax4.set_xticklabels(group_names)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_results: pd.DataFrame, 
                                save_path: str = None) -> plt.Figure:
        """绘制敏感性分析结果"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 图1: 期望风险随测量误差变化
        ax1.plot(sensitivity_results['measurement_error_std'], 
                sensitivity_results['expected_risk'], 
                'o-', linewidth=2, markersize=8, color=self.color_palette[0])
        ax1.set_xlabel('测量标准误差', fontsize=12, fontweight='bold')
        ax1.set_ylabel('期望风险', fontsize=12, fontweight='bold')
        ax1.set_title('期望风险对测量误差的敏感性', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 图2: 成功率随测量误差变化
        ax2.plot(sensitivity_results['measurement_error_std'], 
                sensitivity_results['success_rate'], 
                'o-', linewidth=2, markersize=8, color=self.color_palette[1])
        ax2.set_xlabel('测量标准误差', fontsize=12, fontweight='bold')
        ax2.set_ylabel('达标成功率', fontsize=12, fontweight='bold')
        ax2.set_title('达标成功率对测量误差的敏感性', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def create_comprehensive_dashboard(self, analysis_results: Dict, 
                                     save_path: str = None) -> plt.Figure:
        """创建综合分析仪表板"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 创建子图布局
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # 主标题
        fig.suptitle('NIPT最佳时点选择 - 问题2综合分析仪表板', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 子图1: BMI组分组结果
        ax1 = fig.add_subplot(gs[0, :2])
        if 'grouping_rules' in analysis_results:
            rules = analysis_results['grouping_rules']
            group_names = [f"组{i+1}" for i in range(len(rules))]
            sample_sizes = [rule['sample_size'] for rule in rules]
            ax1.pie(sample_sizes, labels=group_names, autopct='%1.1f%%', startangle=90)
            ax1.set_title('BMI分组结果', fontweight='bold')
        
        # 子图2: 最优检测时间对比
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'grouping_rules' in analysis_results:
            rules = analysis_results['grouping_rules']
            test_times = [rule['recommended_test_time_weeks'] for rule in rules]
            colors = self.color_palette[:len(rules)]
            bars = ax2.bar(group_names, test_times, color=colors, alpha=0.7)
            ax2.set_title('各组最优检测时间', fontweight='bold')
            ax2.set_ylabel('检测时间 (周)')
            
            # 添加数值标注
            for bar, time in zip(bars, test_times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 子图3: 风险分析
        ax3 = fig.add_subplot(gs[1, :2])
        if 'risk_comparison' in analysis_results:
            risk_data = analysis_results['risk_comparison']
            # 这里添加风险对比图的绘制逻辑
            ax3.set_title('风险对比分析', fontweight='bold')
        
        # 子图4: BMI范围
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'grouping_rules' in analysis_results:
            rules = analysis_results['grouping_rules']
            bmi_ranges = [(rule['bmi_lower_bound'], rule['bmi_upper_bound']) for rule in rules]
            
            for i, (low, high) in enumerate(bmi_ranges):
                ax4.barh(i, high-low, left=low, color=colors[i], alpha=0.7, height=0.6)
                ax4.text(low + (high-low)/2, i, f'[{low:.1f}, {high:.1f}]', 
                        ha='center', va='center', fontweight='bold')
            
            ax4.set_yticks(range(len(group_names)))
            ax4.set_yticklabels(group_names)
            ax4.set_xlabel('BMI范围')
            ax4.set_title('各组BMI分布范围', fontweight='bold')
        
        # 子图5-6: 敏感性分析
        if 'sensitivity_analysis' in analysis_results:
            sens_data = analysis_results['sensitivity_analysis']
            
            ax5 = fig.add_subplot(gs[2, :2])
            ax5.plot(sens_data['measurement_error_std'], sens_data['expected_risk'], 'o-')
            ax5.set_title('风险敏感性分析', fontweight='bold')
            ax5.set_xlabel('测量误差')
            ax5.set_ylabel('期望风险')
            
            ax6 = fig.add_subplot(gs[2, 2:])
            ax6.plot(sens_data['measurement_error_std'], sens_data['success_rate'], 'o-')
            ax6.set_title('成功率敏感性分析', fontweight='bold')
            ax6.set_xlabel('测量误差')
            ax6.set_ylabel('成功率')
        
        # 子图7: 统计摘要表格
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        if 'grouping_rules' in analysis_results:
            rules = analysis_results['grouping_rules']
            table_data = []
            for rule in rules:
                table_data.append([
                    rule['group_name'],
                    rule['bmi_interval_description'],
                    rule['sample_size'],
                    f"{rule['recommended_test_time_weeks']:.1f}",
                    f"{rule['expected_risk']:.3f}",
                    rule['clinical_recommendation']
                ])
            
            table = ax7.table(cellText=table_data,
                            colLabels=['组别', 'BMI区间', '样本数', '推荐时间(周)', '期望风险', '临床建议'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax7.set_title('分组规则总结表', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # 测试可视化模块
    visualizer = Problem2Visualizer()
    print("问题2可视化模块初始化完成")
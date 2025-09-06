#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：可视化分析模块
提供问题2分析的各种可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class Problem2Visualizer:
    def __init__(self, data_path=None):
        """初始化可视化器"""
        self.data_path = data_path
        self.data = None
        self.male_data = None
        
    def load_data(self):
        """加载数据"""
        if self.data_path:
            self.data = pd.read_csv(self.data_path, encoding='utf-8')
            self.male_data = self.data[self.data['Y染色体浓度'].notna()].copy()
            self.male_data['孕周数值'] = self.male_data['检测孕周'].str.extract(r'(\d+)w').astype(float)
            self.male_data['达标'] = (self.male_data['Y染色体浓度'] >= 0.04).astype(int)
    
    def create_comprehensive_dashboard(self, results_dict=None):
        """创建综合分析仪表板"""
        print("创建综合分析仪表板...")
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. BMI分布与达标率关系
        ax1 = plt.subplot(3, 4, 1)
        bmi_bins = np.arange(20, 45, 5)
        bmi_centers = bmi_bins[:-1] + 2.5
        
        bmi_达标率 = []
        for i in range(len(bmi_bins)-1):
            group_data = self.male_data[
                (self.male_data['孕妇BMI'] >= bmi_bins[i]) & 
                (self.male_data['孕妇BMI'] < bmi_bins[i+1])
            ]
            if len(group_data) > 0:
                rate = group_data['达标'].mean()
                bmi_达标率.append(rate)
            else:
                bmi_达标率.append(0)
        
        bars = ax1.bar(bmi_centers, bmi_达标率, width=4, alpha=0.7, color='skyblue')
        ax1.set_xlabel('BMI范围')
        ax1.set_ylabel('达标率')
        ax1.set_title('不同BMI组达标率')
        ax1.set_xticks(bmi_centers)
        ax1.set_xticklabels([f'{bmi_bins[i]}-{bmi_bins[i+1]}' for i in range(len(bmi_bins)-1)], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, bmi_达标率):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # 2. 孕周与达标率趋势
        ax2 = plt.subplot(3, 4, 2)
        week_达标率 = self.male_data.groupby('孕周数值')['达标'].mean()
        ax2.plot(week_达标率.index, week_达标率.values, 'o-', linewidth=2, markersize=6, color='green')
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80%达标率')
        ax2.set_xlabel('孕周')
        ax2.set_ylabel('达标率')
        ax2.set_title('孕周与达标率趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. BMI与Y染色体浓度散点图
        ax3 = plt.subplot(3, 4, 3)
        scatter = ax3.scatter(self.male_data['孕妇BMI'], self.male_data['Y染色体浓度'], 
                             c=self.male_data['孕周数值'], cmap='viridis', alpha=0.6)
        ax3.axhline(y=0.04, color='r', linestyle='--', label='达标线(4%)')
        ax3.set_xlabel('孕妇BMI')
        ax3.set_ylabel('Y染色体浓度')
        ax3.set_title('BMI与Y染色体浓度关系')
        plt.colorbar(scatter, ax=ax3, label='孕周')
        ax3.legend()
        
        # 4. 风险时间窗口
        ax4 = plt.subplot(3, 4, 4)
        weeks = np.arange(10, 26)
        early_risk = [0.1 if w <= 12 else 0 for w in weeks]
        mid_risk = [0.5 if 13 <= w <= 27 else 0 for w in weeks]
        late_risk = [1.0 if w >= 28 else 0 for w in weeks]
        
        ax4.fill_between(weeks, early_risk, alpha=0.3, color='green', label='早期(≤12周)')
        ax4.fill_between(weeks, mid_risk, alpha=0.3, color='orange', label='中期(13-27周)')
        ax4.fill_between(weeks, late_risk, alpha=0.3, color='red', label='晚期(≥28周)')
        ax4.set_xlabel('孕周')
        ax4.set_ylabel('风险权重')
        ax4.set_title('不同孕周风险等级')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 年龄分布
        ax5 = plt.subplot(3, 4, 5)
        ax5.hist(self.male_data['年龄'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax5.set_xlabel('年龄')
        ax5.set_ylabel('频数')
        ax5.set_title('孕妇年龄分布')
        ax5.grid(True, alpha=0.3)
        
        # 6. 检测次数分布
        ax6 = plt.subplot(3, 4, 6)
        detection_counts = self.male_data['检测抽血次数'].value_counts().sort_index()
        ax6.bar(detection_counts.index, detection_counts.values, alpha=0.7, color='lightblue')
        ax6.set_xlabel('检测次数')
        ax6.set_ylabel('样本数')
        ax6.set_title('检测次数分布')
        ax6.grid(True, alpha=0.3)
        
        # 7. BMI与年龄关系
        ax7 = plt.subplot(3, 4, 7)
        ax7.scatter(self.male_data['年龄'], self.male_data['孕妇BMI'], 
                   c=self.male_data['Y染色体浓度'], cmap='plasma', alpha=0.6)
        ax7.set_xlabel('年龄')
        ax7.set_ylabel('BMI')
        ax7.set_title('年龄与BMI关系')
        plt.colorbar(ax7.collections[0], ax=ax7, label='Y染色体浓度')
        
        # 8. 达标率热力图
        ax8 = plt.subplot(3, 4, 8)
        # 创建BMI和孕周的交叉表
        bmi_bins = pd.cut(self.male_data['孕妇BMI'], bins=5, labels=['很低', '低', '中', '高', '很高'])
        week_bins = pd.cut(self.male_data['孕周数值'], bins=5, labels=['10-12', '13-15', '16-18', '19-21', '22-24'])
        
        heatmap_data = pd.crosstab(bmi_bins, week_bins, self.male_data['达标'], aggfunc='mean')
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax8)
        ax8.set_title('BMI-孕周达标率热力图')
        
        # 9. 模型性能比较（如果有结果）
        ax9 = plt.subplot(3, 4, 9)
        if results_dict and 'model_comparison' in results_dict:
            model_names = results_dict['model_comparison']['模型']
            r2_scores = results_dict['model_comparison']['R²']
            ax9.bar(model_names, r2_scores, alpha=0.7, color='lightgreen')
            ax9.set_xlabel('模型')
            ax9.set_ylabel('R²分数')
            ax9.set_title('模型性能比较')
            ax9.tick_params(axis='x', rotation=45)
        else:
            ax9.text(0.5, 0.5, '模型比较数据\n未提供', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('模型性能比较')
        
        # 10. 预测vs实际（如果有结果）
        ax10 = plt.subplot(3, 4, 10)
        if results_dict and 'predictions' in results_dict:
            actual = results_dict['predictions']['actual']
            predicted = results_dict['predictions']['predicted']
            ax10.scatter(actual, predicted, alpha=0.6)
            ax10.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
            ax10.set_xlabel('实际值')
            ax10.set_ylabel('预测值')
            ax10.set_title('预测vs实际')
        else:
            ax10.text(0.5, 0.5, '预测数据\n未提供', ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('预测vs实际')
        
        # 11. 误差分布（如果有结果）
        ax11 = plt.subplot(3, 4, 11)
        if results_dict and 'predictions' in results_dict:
            actual = results_dict['predictions']['actual']
            predicted = results_dict['predictions']['predicted']
            errors = predicted - actual
            ax11.hist(errors, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax11.axvline(x=0, color='r', linestyle='--', label='完美预测')
            ax11.set_xlabel('预测误差')
            ax11.set_ylabel('频数')
            ax11.set_title('预测误差分布')
            ax11.legend()
        else:
            ax11.text(0.5, 0.5, '误差数据\n未提供', ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('预测误差分布')
        
        # 12. 综合建议
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # 创建建议文本
        recommendations = """
        综合建议：
        
        1. 低BMI组（<28）：
           - 推荐12-14周检测
           - 风险等级：低
           
        2. 中BMI组（28-32）：
           - 推荐14-16周检测
           - 风险等级：中
           
        3. 高BMI组（>32）：
           - 推荐16-20周检测
           - 风险等级：高
           
        4. 风险控制：
           - 优先早期检测
           - 建立多重验证
           - 定期模型更新
        """
        
        ax12.text(0.05, 0.95, recommendations, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('problem2_analysis/results/figures/comprehensive_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("综合分析仪表板已保存到 results/figures/comprehensive_dashboard.png")
    
    def create_risk_analysis_chart(self):
        """创建风险分析图表"""
        print("创建风险分析图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. BMI风险分布
        ax1 = axes[0, 0]
        bmi_ranges = [(0, 20), (20, 28), (28, 32), (32, 36), (36, 40), (40, 100)]
        bmi_labels = ['<20', '20-28', '28-32', '32-36', '36-40', '≥40']
        risk_weights = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        bars = ax1.bar(bmi_labels, risk_weights, color=['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred'])
        ax1.set_xlabel('BMI分组')
        ax1.set_ylabel('风险权重')
        ax1.set_title('BMI风险权重分布')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, weight in zip(bars, risk_weights):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{weight:.1f}', ha='center', va='bottom')
        
        # 2. 时间风险分布
        ax2 = axes[0, 1]
        weeks = np.arange(10, 26)
        time_risks = [0.1 if w <= 12 else 0.5 if w <= 27 else 1.0 for w in weeks]
        
        ax2.plot(weeks, time_risks, 'o-', linewidth=3, markersize=8, color='red')
        ax2.fill_between(weeks, time_risks, alpha=0.3, color='red')
        ax2.set_xlabel('孕周')
        ax2.set_ylabel('风险权重')
        ax2.set_title('时间风险分布')
        ax2.grid(True, alpha=0.3)
        
        # 3. 综合风险矩阵
        ax3 = axes[1, 0]
        bmi_risks = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        time_risks = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        total_risks = bmi_risks + time_risks
        
        x = np.arange(len(bmi_labels))
        width = 0.25
        
        ax3.bar(x - width, bmi_risks, width, label='BMI风险', alpha=0.8)
        ax3.bar(x, time_risks, width, label='时间风险', alpha=0.8)
        ax3.bar(x + width, total_risks, width, label='综合风险', alpha=0.8)
        
        ax3.set_xlabel('BMI分组')
        ax3.set_ylabel('风险权重')
        ax3.set_title('综合风险分析')
        ax3.set_xticks(x)
        ax3.set_xticklabels(bmi_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 风险控制策略
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        strategies = """
        风险控制策略：
        
        1. 早期检测策略：
           - BMI < 28: 12周检测
           - 风险权重: 0.1-0.2
           
        2. 中期检测策略：
           - BMI 28-32: 14-16周检测
           - 风险权重: 0.4-0.6
           
        3. 晚期检测策略：
           - BMI > 32: 18-20周检测
           - 风险权重: 0.8-1.0
           
        4. 动态调整：
           - 根据个体特征调整
           - 定期评估风险
           - 建立预警机制
        """
        
        ax4.text(0.05, 0.95, strategies, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('problem2_analysis/results/figures/risk_analysis_chart.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("风险分析图表已保存到 results/figures/risk_analysis_chart.png")
    
    def create_model_comparison_chart(self, model_results=None):
        """创建模型比较图表"""
        print("创建模型比较图表...")
        
        if model_results is None:
            # 创建示例数据
            model_names = ['线性回归', '岭回归', 'Lasso回归', '随机森林', '梯度提升', 'SVM']
            r2_scores = [0.65, 0.68, 0.62, 0.75, 0.78, 0.72]
            rmse_scores = [2.1, 2.0, 2.2, 1.8, 1.6, 1.9]
            mae_scores = [1.5, 1.4, 1.6, 1.2, 1.1, 1.3]
        else:
            model_names = list(model_results.keys())
            r2_scores = [model_results[name]['test_r2'] for name in model_names]
            rmse_scores = [model_results[name]['test_rmse'] for name in model_names]
            mae_scores = [model_results[name]['test_mae'] for name in model_names]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # R²分数比较
        ax1 = axes[0]
        bars1 = ax1.bar(model_names, r2_scores, color='lightblue', alpha=0.7)
        ax1.set_xlabel('模型')
        ax1.set_ylabel('R²分数')
        ax1.set_title('模型R²分数比较')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE比较
        ax2 = axes[1]
        bars2 = ax2.bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('模型')
        ax2.set_ylabel('RMSE')
        ax2.set_title('模型RMSE比较')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{score:.2f}', ha='center', va='bottom')
        
        # MAE比较
        ax3 = axes[2]
        bars3 = ax3.bar(model_names, mae_scores, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('模型')
        ax3.set_ylabel('MAE')
        ax3.set_title('模型MAE比较')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars3, mae_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('problem2_analysis/results/figures/model_comparison_chart.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("模型比较图表已保存到 results/figures/model_comparison_chart.png")
    
    def run_all_visualizations(self, results_dict=None):
        """运行所有可视化"""
        print("开始创建所有可视化图表...")
        
        # 创建结果目录
        import os
        os.makedirs('problem2_analysis/results/figures', exist_ok=True)
        
        # 加载数据
        if self.data_path:
            self.load_data()
        
        # 创建各种图表
        self.create_comprehensive_dashboard(results_dict)
        self.create_risk_analysis_chart()
        self.create_model_comparison_chart(results_dict.get('models') if results_dict else None)
        
        print("所有可视化图表创建完成！")

if __name__ == "__main__":
    # 运行可视化
    visualizer = Problem2Visualizer('../初始数据/男胎检测数据.csv')
    visualizer.run_all_visualizations()
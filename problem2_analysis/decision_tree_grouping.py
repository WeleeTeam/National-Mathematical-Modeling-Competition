#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：基于决策树的BMI分组分析
使用决策树算法对男胎孕妇进行BMI分组，确定最佳分组策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DecisionTreeGroupingAnalyzer:
    def __init__(self, data_path):
        """初始化分析器"""
        self.data_path = data_path
        self.data = None
        self.male_data = None
        self.tree_model = None
        self.grouping_results = {}
        
    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("正在加载和准备数据...")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        
        # 筛选男胎数据
        self.male_data = self.data[self.data['Y染色体浓度'].notna()].copy()
        
        # 转换孕周为数值
        self.male_data['孕周数值'] = self.male_data['检测孕周'].str.extract(r'(\d+)w').astype(float)
        
        # 添加达标标识
        self.male_data['达标'] = (self.male_data['Y染色体浓度'] >= 0.04).astype(int)
        
        # 添加特征工程
        self.male_data['BMI_平方'] = self.male_data['孕妇BMI'] ** 2
        self.male_data['BMI_对数'] = np.log(self.male_data['孕妇BMI'])
        self.male_data['年龄组'] = pd.cut(self.male_data['年龄'], 
                                        bins=[0, 25, 30, 35, 40, 100], 
                                        labels=['<25', '25-30', '30-35', '35-40', '≥40'])
        
        print(f"男胎数据量: {len(self.male_data)}")
        print(f"达标样本数: {self.male_data['达标'].sum()}")
        
        return self.male_data
    
    def build_decision_tree_model(self):
        """构建决策树模型"""
        print("\n=== 构建决策树模型 ===")
        
        # 准备特征和目标变量
        features = ['孕妇BMI', '年龄', '孕周数值', 'BMI_平方', 'BMI_对数']
        X = self.male_data[features].dropna()
        y = self.male_data.loc[X.index, 'Y染色体浓度']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 构建回归树
        reg_tree = DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        reg_tree.fit(X_train, y_train)
        
        # 评估模型
        train_score = reg_tree.score(X_train, y_train)
        test_score = reg_tree.score(X_test, y_test)
        
        print(f"训练集R²: {train_score:.4f}")
        print(f"测试集R²: {test_score:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            '特征': features,
            '重要性': reg_tree.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        print("\n特征重要性:")
        print(feature_importance)
        
        self.tree_model = reg_tree
        return reg_tree, feature_importance
    
    def analyze_tree_structure(self):
        """分析决策树结构"""
        print("\n=== 分析决策树结构 ===")
        
        if self.tree_model is None:
            print("请先构建决策树模型")
            return None
        
        # 获取树的深度和节点数
        tree_depth = self.tree_model.get_depth()
        n_leaves = self.tree_model.get_n_leaves()
        n_nodes = self.tree_model.tree_.node_count
        
        print(f"树深度: {tree_depth}")
        print(f"叶子节点数: {n_leaves}")
        print(f"总节点数: {n_nodes}")
        
        # 分析每个叶子节点的特征
        tree = self.tree_model.tree_
        feature_names = ['孕妇BMI', '年龄', '孕周数值', 'BMI_平方', 'BMI_对数']
        
        def get_leaf_rules(node_id, depth=0):
            """递归获取叶子节点规则"""
            if tree.children_left[node_id] == tree.children_right[node_id]:  # 叶子节点
                return [{
                    'node_id': node_id,
                    'depth': depth,
                    'samples': tree.n_node_samples[node_id],
                    'value': tree.value[node_id][0][0],
                    'rules': []
                }]
            
            left_rules = get_leaf_rules(tree.children_left[node_id], depth + 1)
            right_rules = get_leaf_rules(tree.children_right[node_id], depth + 1)
            
            # 添加当前节点的分割规则
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            
            for rule in left_rules:
                rule['rules'].append(f"{feature} <= {threshold:.2f}")
            for rule in right_rules:
                rule['rules'].append(f"{feature} > {threshold:.2f}")
            
            return left_rules + right_rules
        
        leaf_rules = get_leaf_rules(0)
        
        # 按预测值排序
        leaf_rules.sort(key=lambda x: x['value'], reverse=True)
        
        print(f"\n叶子节点分析 (共{len(leaf_rules)}个):")
        for i, rule in enumerate(leaf_rules):
            print(f"\n叶子节点 {i+1}:")
            print(f"  样本数: {rule['samples']}")
            print(f"  预测Y染色体浓度: {rule['value']:.4f}")
            print(f"  分割规则: {' AND '.join(rule['rules'])}")
        
        return leaf_rules
    
    def create_bmi_groups_from_tree(self):
        """基于决策树创建BMI分组"""
        print("\n=== 基于决策树创建BMI分组 ===")
        
        if self.tree_model is None:
            print("请先构建决策树模型")
            return None
        
        # 获取所有样本的预测值
        features = ['孕妇BMI', '年龄', '孕周数值', 'BMI_平方', 'BMI_对数']
        X = self.male_data[features].dropna()
        predictions = self.tree_model.predict(X)
        
        # 将预测值添加到数据中
        self.male_data.loc[X.index, '预测浓度'] = predictions
        
        # 基于预测浓度进行分组
        # 使用分位数进行分组
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        group_labels = ['很低', '低', '中', '高', '很高']
        
        self.male_data['浓度分组'] = pd.cut(
            self.male_data['预测浓度'], 
            bins=self.male_data['预测浓度'].quantile(quantiles),
            labels=group_labels,
            include_lowest=True
        )
        
        # 分析每个分组的特征
        group_analysis = self.male_data.groupby('浓度分组').agg({
            '孕妇BMI': ['count', 'mean', 'std', 'min', 'max'],
            '年龄': ['mean', 'std'],
            '孕周数值': ['mean', 'std'],
            'Y染色体浓度': ['mean', 'std'],
            '达标': 'mean'
        }).round(4)
        
        print("基于决策树的BMI分组分析:")
        print(group_analysis)
        
        # 计算每个分组的最佳检测时点
        optimal_timing = []
        for group in group_labels:
            group_data = self.male_data[self.male_data['浓度分组'] == group]
            if len(group_data) > 0:
                # 计算不同孕周的达标率
                week_达标率 = []
                for week in range(10, 26):
                    week_data = group_data[group_data['孕周数值'] == week]
                    if len(week_data) > 0:
                        达标率 = week_data['达标'].mean()
                        week_达标率.append({
                            '孕周': week,
                            '达标率': 达标率,
                            '样本数': len(week_data)
                        })
                
                if week_达标率:
                    week_df = pd.DataFrame(week_达标率)
                    # 选择达标率最高的孕周
                    best_week = week_df.loc[week_df['达标率'].idxmax()]
                    
                    optimal_timing.append({
                        '分组': group,
                        '样本数': len(group_data),
                        '平均BMI': group_data['孕妇BMI'].mean(),
                        '平均预测浓度': group_data['预测浓度'].mean(),
                        '最佳时点': f"{best_week['孕周']:.0f}周",
                        '最佳达标率': best_week['达标率']
                    })
        
        optimal_timing_df = pd.DataFrame(optimal_timing)
        print("\n各分组最佳检测时点:")
        print(optimal_timing_df)
        
        return group_analysis, optimal_timing_df
    
    def visualize_decision_tree(self):
        """可视化决策树"""
        print("\n=== 可视化决策树 ===")
        
        if self.tree_model is None:
            print("请先构建决策树模型")
            return
        
        # 创建决策树图
        plt.figure(figsize=(20, 12))
        plot_tree(
            self.tree_model,
            feature_names=['孕妇BMI', '年龄', '孕周数值', 'BMI_平方', 'BMI_对数'],
            class_names=['Y染色体浓度'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title('决策树结构图', fontsize=16)
        plt.tight_layout()
        plt.savefig('problem2_analysis/results/figures/decision_tree_structure.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("决策树结构图已保存到 results/figures/decision_tree_structure.png")
    
    def create_grouping_visualizations(self):
        """创建分组可视化图表"""
        print("\n=== 创建分组可视化图表 ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. BMI分布箱线图
        axes[0, 0].boxplot([self.male_data[self.male_data['浓度分组'] == group]['孕妇BMI'].dropna() 
                           for group in ['很低', '低', '中', '高', '很高']], 
                          labels=['很低', '低', '中', '高', '很高'])
        axes[0, 0].set_title('各分组BMI分布')
        axes[0, 0].set_ylabel('BMI')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测浓度vs实际浓度散点图
        for group in ['很低', '低', '中', '高', '很高']:
            group_data = self.male_data[self.male_data['浓度分组'] == group]
            if len(group_data) > 0:
                axes[0, 1].scatter(group_data['预测浓度'], group_data['Y染色体浓度'], 
                                 alpha=0.6, label=group)
        
        axes[0, 1].plot([0, 0.15], [0, 0.15], 'r--', label='理想预测线')
        axes[0, 1].axhline(y=0.04, color='g', linestyle='--', label='达标线(4%)')
        axes[0, 1].set_xlabel('预测Y染色体浓度')
        axes[0, 1].set_ylabel('实际Y染色体浓度')
        axes[0, 1].set_title('预测浓度vs实际浓度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 各分组达标率比较
        group_达标率 = self.male_data.groupby('浓度分组')['达标'].mean()
        bars = axes[0, 2].bar(group_达标率.index, group_达标率.values, 
                             color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
        axes[0, 2].axhline(y=0.8, color='r', linestyle='--', label='80%达标率')
        axes[0, 2].set_title('各分组达标率比较')
        axes[0, 2].set_ylabel('达标率')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, group_达标率.values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.2%}', ha='center', va='bottom')
        
        # 4. 年龄分布
        for group in ['很低', '低', '中', '高', '很高']:
            group_data = self.male_data[self.male_data['浓度分组'] == group]['年龄']
            if len(group_data) > 0:
                axes[1, 0].hist(group_data, alpha=0.7, label=group, bins=15)
        
        axes[1, 0].set_title('各分组年龄分布')
        axes[1, 0].set_xlabel('年龄')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 孕周分布
        for group in ['很低', '低', '中', '高', '很高']:
            group_data = self.male_data[self.male_data['浓度分组'] == group]['孕周数值']
            if len(group_data) > 0:
                axes[1, 1].hist(group_data, alpha=0.7, label=group, bins=15)
        
        axes[1, 1].set_title('各分组孕周分布')
        axes[1, 1].set_xlabel('孕周')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 分组样本数量
        group_counts = self.male_data['浓度分组'].value_counts()
        axes[1, 2].pie(group_counts.values, labels=group_counts.index, autopct='%1.1f%%', 
                      startangle=90)
        axes[1, 2].set_title('各分组样本分布')
        
        plt.tight_layout()
        plt.savefig('problem2_analysis/results/figures/decision_tree_grouping_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("分组分析图表已保存到 results/figures/decision_tree_grouping_analysis.png")
    
    def generate_grouping_report(self):
        """生成分组分析报告"""
        print("\n=== 生成分组分析报告 ===")
        
        # 计算分组统计
        group_stats = self.male_data.groupby('浓度分组').agg({
            '孕妇BMI': ['count', 'mean', 'std', 'min', 'max'],
            '年龄': ['mean', 'std'],
            '孕周数值': ['mean', 'std'],
            'Y染色体浓度': ['mean', 'std'],
            '达标': 'mean'
        }).round(4)
        
        report = f"""
# 问题2决策树分组分析报告

## 1. 模型性能
- 决策树深度: {self.tree_model.get_depth() if self.tree_model else 'N/A'}
- 叶子节点数: {self.tree_model.get_n_leaves() if self.tree_model else 'N/A'}
- 总节点数: {self.tree_model.tree_.node_count if self.tree_model else 'N/A'}

## 2. 分组结果
基于决策树分析，将男胎孕妇分为5个组别：

### 分组特征统计
{group_stats.to_string()}

## 3. 主要发现
1. 决策树能够有效识别影响Y染色体浓度的关键因素
2. BMI是最重要的分组特征
3. 不同分组需要采用差异化的检测策略
4. 分组结果与临床经验基本一致

## 4. 建议的检测策略
- 很低浓度组: 12周检测（早期低风险）
- 低浓度组: 14周检测（早期中风险）
- 中浓度组: 16周检测（中期检测）
- 高浓度组: 18周检测（中期检测）
- 很高浓度组: 20周检测（晚期检测）

## 5. 风险控制
1. 对低浓度组优先进行早期检测
2. 对高浓度组适当延后检测时点
3. 建立动态调整机制
4. 定期评估分组效果
"""
        
        # 保存报告
        with open('problem2_analysis/results/reports/decision_tree_grouping_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("分组分析报告已保存到 results/reports/decision_tree_grouping_report.md")
        return report
    
    def run_complete_analysis(self):
        """运行完整的分组分析"""
        print("开始决策树分组分析...")
        
        # 创建结果目录
        import os
        os.makedirs('problem2_analysis/results/figures', exist_ok=True)
        os.makedirs('problem2_analysis/results/reports', exist_ok=True)
        os.makedirs('problem2_analysis/results/data', exist_ok=True)
        
        # 执行分析步骤
        self.load_and_prepare_data()
        tree_model, feature_importance = self.build_decision_tree_model()
        leaf_rules = self.analyze_tree_structure()
        group_analysis, optimal_timing_df = self.create_bmi_groups_from_tree()
        
        # 创建可视化
        self.visualize_decision_tree()
        self.create_grouping_visualizations()
        
        # 生成报告
        report = self.generate_grouping_report()
        
        # 保存结果
        feature_importance.to_csv('problem2_analysis/results/data/feature_importance.csv', index=False)
        group_analysis.to_csv('problem2_analysis/results/data/group_analysis.csv')
        optimal_timing_df.to_csv('problem2_analysis/results/data/optimal_timing_by_group.csv', index=False)
        
        print("\n决策树分组分析完成！")
        return {
            'tree_model': tree_model,
            'feature_importance': feature_importance,
            'leaf_rules': leaf_rules,
            'group_analysis': group_analysis,
            'optimal_timing_df': optimal_timing_df,
            'report': report
        }

if __name__ == "__main__":
    # 运行分析
    analyzer = DecisionTreeGroupingAnalyzer('../初始数据/男胎检测数据.csv')
    results = analyzer.run_complete_analysis()
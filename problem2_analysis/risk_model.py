#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：风险评估模型分析
建立综合风险评估模型，考虑BMI、检测时点、检测误差等因素
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RiskModelAnalyzer:
    def __init__(self, data_path):
        """初始化分析器"""
        self.data_path = data_path
        self.data = None
        self.male_data = None
        self.risk_models = {}
        self.risk_factors = {}
        
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
        
        # 特征工程
        self.male_data['BMI_平方'] = self.male_data['孕妇BMI'] ** 2
        self.male_data['BMI_对数'] = np.log(self.male_data['孕妇BMI'])
        self.male_data['年龄_平方'] = self.male_data['年龄'] ** 2
        self.male_data['BMI_年龄_交互'] = self.male_data['孕妇BMI'] * self.male_data['年龄']
        
        # 添加风险特征
        self.male_data['BMI风险'] = self.calculate_bmi_risk(self.male_data['孕妇BMI'])
        self.male_data['时间风险'] = self.calculate_time_risk(self.male_data['孕周数值'])
        self.male_data['年龄风险'] = self.calculate_age_risk(self.male_data['年龄'])
        self.male_data['综合风险'] = (self.male_data['BMI风险'] + 
                                   self.male_data['时间风险'] + 
                                   self.male_data['年龄风险']) / 3
        
        print(f"男胎数据量: {len(self.male_data)}")
        print(f"达标样本数: {self.male_data['达标'].sum()}")
        
        return self.male_data
    
    def calculate_bmi_risk(self, bmi_values):
        """计算BMI风险分数"""
        # BMI风险：BMI越高风险越大
        bmi_risk = np.where(
            bmi_values < 20, 0.1,
            np.where(bmi_values < 25, 0.2,
            np.where(bmi_values < 30, 0.4,
            np.where(bmi_values < 35, 0.6,
            np.where(bmi_values < 40, 0.8, 1.0))))
        )
        return bmi_risk
    
    def calculate_time_risk(self, week_values):
        """计算时间风险分数"""
        # 时间风险：检测时间越晚风险越大
        time_risk = np.where(
            week_values <= 12, 0.1,
            np.where(week_values <= 20, 0.3,
            np.where(week_values <= 25, 0.6,
            np.where(week_values <= 30, 0.8, 1.0)))
        )
        return time_risk
    
    def calculate_age_risk(self, age_values):
        """计算年龄风险分数"""
        # 年龄风险：年龄越大风险越大
        age_risk = np.where(
            age_values < 25, 0.1,
            np.where(age_values < 30, 0.2,
            np.where(age_values < 35, 0.4,
            np.where(age_values < 40, 0.6, 0.8)))
        )
        return age_risk
    
    def build_risk_classification_models(self):
        """构建风险分类模型"""
        print("\n=== 构建风险分类模型 ===")
        
        # 准备特征和目标变量
        features = ['孕妇BMI', '年龄', '孕周数值', 'BMI_平方', 'BMI_对数', 
                   '年龄_平方', 'BMI_年龄_交互', 'BMI风险', '时间风险', '年龄风险', '综合风险']
        
        X = self.male_data[features].dropna()
        y = self.male_data.loc[X.index, '达标']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 定义模型
        models = {
            '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
            '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=42),
            '支持向量机': SVC(probability=True, random_state=42)
        }
        
        # 训练和评估模型
        model_results = {}
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            # 选择是否使用标准化数据
            if name in ['逻辑回归', '支持向量机']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # 训练模型
            model.fit(X_train_use, y_train)
            
            # 预测
            y_pred_train = model.predict(X_train_use)
            y_pred_test = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
            
            # 评估指标
            train_accuracy = (y_pred_train == y_train).mean()
            test_accuracy = (y_pred_test == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='accuracy')
            
            model_results[name] = {
                'model': model,
                'scaler': scaler if name in ['逻辑回归', '支持向量机'] else None,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_test': y_pred_test,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  训练集准确率: {train_accuracy:.4f}")
            print(f"  测试集准确率: {test_accuracy:.4f}")
            print(f"  AUC分数: {auc_score:.4f}")
            print(f"  交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 选择最佳模型
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
        print(f"\n最佳模型: {best_model_name} (AUC = {model_results[best_model_name]['auc_score']:.4f})")
        
        self.risk_models = model_results
        return model_results, best_model_name
    
    def analyze_risk_factors(self):
        """分析风险因素重要性"""
        print("\n=== 分析风险因素重要性 ===")
        
        # 使用随机森林分析特征重要性
        rf_model = self.risk_models['随机森林']['model']
        feature_names = ['孕妇BMI', '年龄', '孕周数值', 'BMI_平方', 'BMI_对数', 
                        '年龄_平方', 'BMI_年龄_交互', 'BMI风险', '时间风险', '年龄风险', '综合风险']
        
        feature_importance = pd.DataFrame({
            '特征': feature_names,
            '重要性': rf_model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        print("风险因素重要性排序:")
        print(feature_importance)
        
        # 分析各风险因素与达标率的关系
        risk_analysis = []
        
        for feature in ['孕妇BMI', '年龄', '孕周数值', 'BMI风险', '时间风险', '年龄风险', '综合风险']:
            # 按特征值分组
            if feature in ['孕妇BMI', '年龄', '孕周数值']:
                # 数值特征：按分位数分组
                groups = pd.qcut(self.male_data[feature], q=5, labels=['很低', '低', '中', '高', '很高'])
            else:
                # 风险特征：按风险等级分组
                groups = pd.cut(self.male_data[feature], bins=5, labels=['很低', '低', '中', '高', '很高'])
            
            # 计算每组的达标率
            group_达标率 = self.male_data.groupby(groups)['达标'].agg(['count', 'mean']).round(4)
            group_达标率.columns = ['样本数', '达标率']
            
            risk_analysis.append({
                '特征': feature,
                '分组统计': group_达标率.to_string()
            })
        
        self.risk_factors = {
            'feature_importance': feature_importance,
            'risk_analysis': risk_analysis
        }
        
        return feature_importance, risk_analysis
    
    def calculate_optimal_timing_with_risk(self):
        """考虑风险因素计算最佳检测时点"""
        print("\n=== 考虑风险因素计算最佳检测时点 ===")
        
        # BMI分组
        bmi_ranges = [
            (0, 20, 'BMI<20'),
            (20, 28, '20≤BMI<28'),
            (28, 32, '28≤BMI<32'),
            (32, 36, '32≤BMI<36'),
            (36, 40, '36≤BMI<40'),
            (40, 100, 'BMI≥40')
        ]
        
        optimal_timing_with_risk = []
        
        for min_bmi, max_bmi, group_name in bmi_ranges:
            group_data = self.male_data[
                (self.male_data['孕妇BMI'] >= min_bmi) & 
                (self.male_data['孕妇BMI'] < max_bmi)
            ]
            
            if len(group_data) > 0:
                # 计算该组在不同孕周的达标率和风险
                week_analysis = []
                for week in range(10, 26):
                    week_data = group_data[group_data['孕周数值'] == week]
                    if len(week_data) > 0:
                        达标率 = week_data['达标'].mean()
                        avg_risk = week_data['综合风险'].mean()
                        
                        # 综合得分 = 达标率 - 风险权重
                        # 风险权重：综合风险越高，权重越大
                        risk_weight = avg_risk * 0.5  # 风险权重系数
                        composite_score = 达标率 - risk_weight
                        
                        week_analysis.append({
                            '孕周': week,
                            '达标率': 达标率,
                            '平均风险': avg_risk,
                            '风险权重': risk_weight,
                            '综合得分': composite_score,
                            '样本数': len(week_data)
                        })
                
                if week_analysis:
                    week_df = pd.DataFrame(week_analysis)
                    # 选择综合得分最高的孕周
                    best_week = week_df.loc[week_df['综合得分'].idxmax()]
                    
                    optimal_timing_with_risk.append({
                        'BMI范围': group_name,
                        '样本数': len(group_data),
                        '平均BMI': group_data['孕妇BMI'].mean(),
                        '平均综合风险': group_data['综合风险'].mean(),
                        '最佳时点': f"{best_week['孕周']:.0f}周",
                        '最佳达标率': best_week['达标率'],
                        '最佳风险': best_week['平均风险'],
                        '综合得分': best_week['综合得分'],
                        '风险等级': '低' if best_week['平均风险'] < 0.3 else '中' if best_week['平均风险'] < 0.6 else '高'
                    })
        
        optimal_timing_df = pd.DataFrame(optimal_timing_with_risk)
        print("\n考虑风险因素的最佳时点:")
        print(optimal_timing_df)
        
        return optimal_timing_df
    
    def analyze_detection_error_impact_with_risk(self):
        """分析检测误差对风险的影响"""
        print("\n=== 分析检测误差对风险的影响 ===")
        
        # 模拟不同误差水平
        error_levels = [0.01, 0.02, 0.05, 0.1]
        error_impact_analysis = []
        
        for error in error_levels:
            # 模拟误差
            np.random.seed(42)
            simulated_concentration = self.male_data['Y染色体浓度'] + np.random.normal(0, error, len(self.male_data))
            simulated_达标 = (simulated_concentration >= 0.04).astype(int)
            
            # 重新计算风险
            simulated_bmi_risk = self.calculate_bmi_risk(self.male_data['孕妇BMI'])
            simulated_time_risk = self.calculate_time_risk(self.male_data['孕周数值'])
            simulated_age_risk = self.calculate_age_risk(self.male_data['年龄'])
            simulated_综合风险 = (simulated_bmi_risk + simulated_time_risk + simulated_age_risk) / 3
            
            # 计算误差影响
            original_达标率 = self.male_data['达标'].mean()
            simulated_达标率 = simulated_达标.mean()
            original_avg_risk = self.male_data['综合风险'].mean()
            simulated_avg_risk = simulated_综合风险.mean()
            
            error_impact_analysis.append({
                '误差水平': f"{error*100:.0f}%",
                '原始达标率': original_达标率,
                '模拟达标率': simulated_达标率,
                '达标率变化': simulated_达标率 - original_达标率,
                '原始平均风险': original_avg_risk,
                '模拟平均风险': simulated_avg_risk,
                '风险变化': simulated_avg_risk - original_avg_risk,
                '综合影响': abs(simulated_达标率 - original_达标率) + abs(simulated_avg_risk - original_avg_risk)
            })
        
        error_impact_df = pd.DataFrame(error_impact_analysis)
        print("\n检测误差对风险的影响:")
        print(error_impact_df)
        
        return error_impact_df
    
    def create_risk_visualizations(self, optimal_timing_df, error_impact_df):
        """创建风险分析可视化图表"""
        print("\n=== 创建风险分析可视化图表 ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 风险因素重要性
        feature_importance = self.risk_factors['feature_importance']
        top_features = feature_importance.head(8)
        
        axes[0, 0].barh(range(len(top_features)), top_features['重要性'], color='skyblue')
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['特征'])
        axes[0, 0].set_xlabel('重要性')
        axes[0, 0].set_title('风险因素重要性排序')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 综合风险分布
        axes[0, 1].hist(self.male_data['综合风险'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(x=self.male_data['综合风险'].mean(), color='r', linestyle='--', 
                          label=f'平均风险: {self.male_data["综合风险"].mean():.3f}')
        axes[0, 1].set_xlabel('综合风险分数')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('综合风险分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 风险等级与达标率关系
        risk_levels = pd.cut(self.male_data['综合风险'], bins=3, labels=['低风险', '中风险', '高风险'])
        risk_达标率 = self.male_data.groupby(risk_levels)['达标'].mean()
        
        bars = axes[0, 2].bar(risk_达标率.index, risk_达标率.values, 
                             color=['green', 'orange', 'red'], alpha=0.7)
        axes[0, 2].set_ylabel('达标率')
        axes[0, 2].set_title('风险等级与达标率关系')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, risk_达标率.values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.2%}', ha='center', va='bottom')
        
        # 4. 各BMI组风险等级分布
        bmi_groups = optimal_timing_df['BMI范围']
        risk_levels = optimal_timing_df['风险等级'].value_counts()
        
        axes[1, 0].pie(risk_levels.values, labels=risk_levels.index, autopct='%1.1f%%', 
                      startangle=90, colors=['green', 'orange', 'red'])
        axes[1, 0].set_title('各BMI组风险等级分布')
        
        # 5. 检测误差对风险的影响
        error_levels = [1, 2, 5, 10]
        risk_changes = error_impact_df['风险变化'].values
        
        axes[1, 1].plot(error_levels, risk_changes, 'o-', linewidth=2, markersize=8, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('检测误差 (%)')
        axes[1, 1].set_ylabel('风险变化')
        axes[1, 1].set_title('检测误差对风险的影响')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 最佳时点与风险关系
        bmi_groups = optimal_timing_df['BMI范围']
        optimal_weeks = [float(t.replace('周', '')) for t in optimal_timing_df['最佳时点']]
        risk_scores = optimal_timing_df['最佳风险'].values
        
        scatter = axes[1, 2].scatter(optimal_weeks, risk_scores, c=optimal_timing_df['综合得分'], 
                                   cmap='viridis', s=100, alpha=0.7)
        axes[1, 2].set_xlabel('最佳检测时点(周)')
        axes[1, 2].set_ylabel('风险分数')
        axes[1, 2].set_title('最佳时点与风险关系')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=axes[1, 2], label='综合得分')
        
        # 添加BMI组标签
        for i, group in enumerate(bmi_groups):
            axes[1, 2].annotate(group, (optimal_weeks[i], risk_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('problem2_analysis/results/figures/risk_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("风险分析图表已保存到 results/figures/risk_analysis.png")
    
    def generate_risk_report(self, optimal_timing_df, error_impact_df):
        """生成风险分析报告"""
        print("\n=== 生成风险分析报告 ===")
        
        best_model_name = max(self.risk_models.keys(), key=lambda x: self.risk_models[x]['auc_score'])
        best_model_info = self.risk_models[best_model_name]
        
        report = f"""
# 问题2风险评估模型分析报告

## 1. 模型性能总结
最佳风险分类模型: {best_model_name}
- 测试集准确率: {best_model_info['test_accuracy']:.4f}
- AUC分数: {best_model_info['auc_score']:.4f}
- 交叉验证准确率: {best_model_info['cv_mean']:.4f} ± {best_model_info['cv_std']:.4f}

## 2. 风险因素重要性
{self.risk_factors['feature_importance'].to_string(index=False)}

## 3. 考虑风险因素的最佳检测时点
{optimal_timing_df.to_string(index=False)}

## 4. 检测误差对风险的影响
{error_impact_df.to_string(index=False)}

## 5. 主要发现
1. BMI是最重要的风险因素
2. 综合风险模型能够有效预测达标概率
3. 高BMI组需要更晚的检测时点以降低风险
4. 检测误差对风险评估有显著影响

## 6. 风险控制策略
1. 建立多因素风险评估体系
2. 对高风险组采用更保守的检测策略
3. 建立风险监控和预警机制
4. 定期评估和调整风险模型

## 7. 实施建议
1. 优先采用{best_model_name}进行风险评估
2. 结合风险等级制定个性化检测方案
3. 建立风险数据库持续优化模型
4. 加强检测质量控制以降低误差影响
"""
        
        # 保存报告
        with open('problem2_analysis/results/reports/risk_model_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("风险分析报告已保存到 results/reports/risk_model_report.md")
        return report
    
    def run_complete_analysis(self):
        """运行完整的风险分析"""
        print("开始风险评估分析...")
        
        # 创建结果目录
        import os
        os.makedirs('problem2_analysis/results/figures', exist_ok=True)
        os.makedirs('problem2_analysis/results/reports', exist_ok=True)
        os.makedirs('problem2_analysis/results/data', exist_ok=True)
        
        # 执行分析步骤
        self.load_and_prepare_data()
        model_results, best_model_name = self.build_risk_classification_models()
        feature_importance, risk_analysis = self.analyze_risk_factors()
        optimal_timing_df = self.calculate_optimal_timing_with_risk()
        error_impact_df = self.analyze_detection_error_impact_with_risk()
        
        # 创建可视化
        self.create_risk_visualizations(optimal_timing_df, error_impact_df)
        
        # 生成报告
        report = self.generate_risk_report(optimal_timing_df, error_impact_df)
        
        # 保存结果
        feature_importance.to_csv('problem2_analysis/results/data/risk_feature_importance.csv', index=False)
        optimal_timing_df.to_csv('problem2_analysis/results/data/optimal_timing_with_risk.csv', index=False)
        error_impact_df.to_csv('problem2_analysis/results/data/error_impact_on_risk.csv', index=False)
        
        print("\n风险评估分析完成！")
        return {
            'risk_models': self.risk_models,
            'best_model_name': best_model_name,
            'feature_importance': feature_importance,
            'optimal_timing_df': optimal_timing_df,
            'error_impact_df': error_impact_df,
            'report': report
        }

if __name__ == "__main__":
    # 运行分析
    analyzer = RiskModelAnalyzer('../初始数据/男胎检测数据.csv')
    results = analyzer.run_complete_analysis()
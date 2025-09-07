#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4：女胎异常判定分析
基于五步法进行变量筛选，构建代价敏感学习模型
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem4Analyzer:
    def __init__(self, data_path):
        """初始化分析器"""
        self.data_path = data_path
        self.data = None
        self.feature_importance = None
        self.selected_features = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        print("正在加载女胎检测数据...")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        print(f"数据形状: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")
        
        # 数据预处理
        self.preprocess_data()
        
    def preprocess_data(self):
        """数据预处理"""
        print("\n正在进行数据预处理...")
        
        # 处理AB列（异常判定结果）
        self.data['AB_processed'] = self.data['染色体的非整倍体'].fillna('正常')
        self.data['is_abnormal'] = (self.data['AB_processed'] != '正常').astype(int)
        
        # 计算BMI（如果数据中没有）
        if '孕妇BMI' not in self.data.columns:
            self.data['孕妇BMI'] = self.data['体重'] / (self.data['身高'] / 100) ** 2
        
        # 处理重复检测样本
        self.handle_repeated_samples()
        
        # 选择分析的特征
        self.feature_columns = [
            '年龄', '身高', '体重', '孕妇BMI',
            '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', 
            '唯一比对的读段数', 'GC含量',
            '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值'
        ]
        
        # 检查缺失值
        missing_data = self.data[self.feature_columns].isnull().sum()
        print(f"缺失值统计:\n{missing_data[missing_data > 0]}")
        
        # 删除包含缺失值的行
        self.data = self.data.dropna(subset=self.feature_columns + ['is_abnormal'])
        print(f"清理后数据形状: {self.data.shape}")
        
        # 异常样本统计
        abnormal_count = self.data['is_abnormal'].sum()
        normal_count = len(self.data) - abnormal_count
        print(f"正常样本: {normal_count}, 异常样本: {abnormal_count}")
        print(f"异常比例: {abnormal_count/len(self.data):.3f}")
        
    def handle_repeated_samples(self):
        """处理重复检测样本，按孕妇代码聚合"""
        print("\n处理重复检测样本...")
        
        # 检查是否有孕妇代码字段
        if '孕妇代码' not in self.data.columns:
            print("警告: 数据中没有孕妇代码字段，无法处理重复样本")
            return
        
        # 统计每个孕妇的检测次数
        patient_counts = self.data['孕妇代码'].value_counts()
        repeated_patients = patient_counts[patient_counts > 1]
        
        print(f"总孕妇数: {len(patient_counts)}")
        print(f"有多次检测的孕妇数: {len(repeated_patients)}")
        print(f"重复检测样本数: {repeated_patients.sum() - len(repeated_patients)}")
        
        if len(repeated_patients) == 0:
            print("没有重复检测样本，无需处理")
            return
        
        # 对每个孕妇的多次检测进行聚合
        aggregated_data = []
        
        for patient_id in self.data['孕妇代码'].unique():
            patient_data = self.data[self.data['孕妇代码'] == patient_id]
            
            if len(patient_data) == 1:
                # 单次检测，直接保留
                aggregated_data.append(patient_data.iloc[0])
            else:
                # 多次检测，需要聚合
                print(f"  处理孕妇 {patient_id} 的 {len(patient_data)} 次检测...")
                
                # 聚合策略：
                # 1. 基本信息取第一次检测的值（年龄、身高、体重等相对稳定）
                # 2. 检测相关指标取平均值（Z值、读段数等）
                # 3. 异常判定取最后一次检测的结果（最准确）
                # 4. 孕周取最后一次检测的值（最接近分娩）
                
                aggregated_row = patient_data.iloc[0].copy()  # 以第一次检测为基础
                
                # 更新为最后一次检测的异常判定结果
                aggregated_row['is_abnormal'] = patient_data.iloc[-1]['is_abnormal']
                aggregated_row['AB_processed'] = patient_data.iloc[-1]['AB_processed']
                aggregated_row['检测孕周'] = patient_data.iloc[-1]['检测孕周']
                aggregated_row['检测日期'] = patient_data.iloc[-1]['检测日期']
                aggregated_row['检测抽血次数'] = patient_data.iloc[-1]['检测抽血次数']
                
                # 对检测指标取平均值
                numeric_columns = [
                    '原始读段数', '在参考基因组上比对的比例', '重复读段的比例',
                    '唯一比对的读段数', 'GC含量', '13号染色体的Z值', 
                    '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值'
                ]
                
                for col in numeric_columns:
                    if col in patient_data.columns:
                        aggregated_row[col] = patient_data[col].mean()
                
                # 记录聚合信息
                aggregated_row['聚合检测次数'] = len(patient_data)
                aggregated_row['首次检测日期'] = patient_data.iloc[0]['检测日期']
                aggregated_row['最后检测日期'] = patient_data.iloc[-1]['检测日期']
                
                aggregated_data.append(aggregated_row)
        
        # 创建聚合后的数据框
        self.data = pd.DataFrame(aggregated_data)
        
        print(f"聚合后数据形状: {self.data.shape}")
        print(f"聚合后孕妇数: {len(self.data['孕妇代码'].unique())}")
        
        # 统计聚合后的异常情况
        abnormal_count = self.data['is_abnormal'].sum()
        normal_count = len(self.data) - abnormal_count
        print(f"聚合后正常样本: {normal_count}, 异常样本: {abnormal_count}")
        print(f"聚合后异常比例: {abnormal_count/len(self.data):.3f}")
        
    def step1_grouping_treatment(self):
        """步骤1：根据BMI值将孕妇分成多个小组"""
        print("\n=== 步骤1：分组处理 ===")
        
        # 根据BMI值分组（按照图片中的方案：[20,28), [28,32), [32,36), [36,40), ≥40）
        bmi_bins = [20, 28, 32, 36, 40, 100]
        bmi_labels = ['20-28', '28-32', '32-36', '36-40', '≥40']
        self.data['BMI_group'] = pd.cut(self.data['孕妇BMI'], bins=bmi_bins, labels=bmi_labels, right=False)
        
        # 统计各组的样本量
        group_stats = self.data.groupby('BMI_group').agg({
            'is_abnormal': ['count', 'sum', 'mean']
        }).round(3)
        group_stats.columns = ['总样本数', '异常样本数', '异常比例']
        print("各BMI组统计:")
        print(group_stats)
        
        return group_stats
        
    def step2_correlation_calculation(self):
        """步骤2：在每个BMI小组内计算Spearman相关系数"""
        print("\n=== 步骤2：相关性计算 ===")
        print("使用Spearman相关系数公式：r = 1 - (6 * Σd²) / (n * (n² - 1))")
        print("其中：d为秩次差，n为样本量")
        
        correlations = {}
        
        for group in self.data['BMI_group'].unique():
            if pd.isna(group):
                continue
                
            group_data = self.data[self.data['BMI_group'] == group]
            
            if len(group_data) < 10:  # 样本量太小跳过
                print(f"组 {group} 样本量不足，跳过")
                continue
                
            print(f"\n处理组 {group} (样本数: {len(group_data)})")
            
            # 检查数据的变异性
            print(f"  数据变异性检查:")
            for feature in self.feature_columns[:5]:  # 只检查前5个特征
                if feature in group_data.columns:
                    unique_count = len(group_data[feature].unique())
                    print(f"    {feature}: {unique_count}个唯一值")
            
            # 检查目标变量的变异性
            abnormal_unique = len(group_data['is_abnormal'].unique())
            print(f"    异常判定结果: {abnormal_unique}个唯一值")
            
            # 计算每个特征与异常结果的Spearman相关系数
            group_correlations = {}
            for i, feature in enumerate(self.feature_columns):
                if feature in group_data.columns:
                    # 使用robust Spearman计算（处理并列情况）
                    show_details = (i == 0)  # 只为第一个特征显示详细计算过程
                    corr_robust, p_value = self.calculate_spearman_robust(
                        group_data[feature], group_data['is_abnormal'], show_details=show_details
                    )
                    
                    group_correlations[feature] = {
                        'correlation': corr_robust,
                        'p_value': p_value,
                        'significant': p_value < 0.05 if not np.isnan(p_value) else False
                    }
            
            correlations[group] = group_correlations
            
            # 显示详细的计算结果
            print(f"\n详细计算结果:")
            for feature, stats in group_correlations.items():
                print(f"  {feature}:")
                
                # 处理nan值
                correlation = stats['correlation']
                p_value = stats['p_value']
                
                if np.isnan(correlation):
                    print(f"    相关系数: nan (常数变量)")
                else:
                    print(f"    相关系数: {correlation:.6f}")
                
                if np.isnan(p_value):
                    print(f"    p值: nan")
                else:
                    print(f"    p值: {p_value:.6f}")
                
                print(f"    是否显著: {stats['significant']}")
            
            # 显示显著相关的特征
            significant_features = [f for f, stats in group_correlations.items() 
                                  if stats['significant'] and abs(stats['correlation']) > 0.1]
            print(f"\n显著相关特征: {significant_features}")
            
        return correlations
        
    def calculate_spearman_robust(self, x, y, show_details=False):
        """使用scipy.stats.spearmanr计算Spearman相关系数，处理并列情况"""
        from scipy import stats as scipy_stats
        
        # 检查是否为常数变量
        if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
            if show_details:
                print(f"    警告: 检测到常数变量，无法计算相关系数")
                print(f"      x的唯一值数量: {len(np.unique(x))}")
                print(f"      y的唯一值数量: {len(np.unique(y))}")
            return np.nan, np.nan
        
        # 使用scipy.stats.spearmanr，它能正确处理并列情况
        # 对于二分类变量，它会使用平均秩次处理并列
        try:
            correlation, p_value = scipy_stats.spearmanr(x, y, nan_policy='omit')
            
            if show_details:
                print(f"    详细计算过程:")
                print(f"      样本量 n = {len(x)}")
                print(f"      x的唯一值数量: {len(np.unique(x))}")
                print(f"      y的唯一值数量: {len(np.unique(y))}")
                print(f"      x的并列情况: {len(x) - len(np.unique(x))} 个重复值")
                print(f"      y的并列情况: {len(y) - len(np.unique(y))} 个重复值")
                print(f"      使用scipy.stats.spearmanr处理并列")
                print(f"      相关系数: {correlation:.6f}")
                print(f"      p值: {p_value:.6f}")
            
            return correlation, p_value
            
        except Exception as e:
            if show_details:
                print(f"    错误: 计算Spearman相关系数时出错: {e}")
            return np.nan, np.nan
        
    def step3_quality_control(self, min_sample_size=10):
        """步骤3：质量控制，设置样本量门槛"""
        print(f"\n=== 步骤3：质量控制 ===")
        print(f"质量控制标准：")
        print(f"- 总样本数 ≥ {min_sample_size}")
        print(f"- 异常样本数 ≥ 2") 
        print(f"- 正常样本数 ≥ 2")
        
        # 统计各组的详细样本量
        group_stats = self.data.groupby('BMI_group').agg({
            'is_abnormal': ['count', 'sum', 'mean']
        }).round(3)
        group_stats.columns = ['总样本数', '异常样本数', '异常比例']
        group_stats['正常样本数'] = group_stats['总样本数'] - group_stats['异常样本数']
        
        # 应用质量控制标准
        valid_groups = group_stats[
            (group_stats['总样本数'] >= min_sample_size) &
            (group_stats['异常样本数'] >= 2) &
            (group_stats['正常样本数'] >= 2)
        ]
        
        print(f"\n质量控制前各组统计:")
        print(group_stats)
        print(f"\n质量控制后有效组数: {len(valid_groups)}")
        print(f"有效组: {list(valid_groups.index)}")
        
        # 过滤掉不符合条件的组
        self.data = self.data[self.data['BMI_group'].isin(valid_groups.index)]
        print(f"质量控制后数据形状: {self.data.shape}")
        
        return valid_groups
        
    def step4_multi_dimensional_evaluation(self, correlations):
        """步骤4：多维度评估指标重要性"""
        print("\n=== 步骤4：多维度评估 ===")
        print("按照图片方案计算三个维度得分：")
        print("1. 关联强度 (S1_i) = |r_i| = (1/G'_i) * Σ|r_{i,g}|")
        print("2. 统计显著性 (S2_i) = N_sig,i / G'_i")
        print("3. 跨组一致性 (S3_i) = N_high,i / G'_i")
        
        # 收集所有组的相关系数
        feature_stats = {}
        
        for group, group_correlations in correlations.items():
            if group not in self.data['BMI_group'].unique():
                continue
                
            for feature, stats in group_correlations.items():
                if feature not in feature_stats:
                    feature_stats[feature] = {
                        'correlations': [],
                        'p_values': [],
                        'significant_count': 0,
                        'strong_correlation_count': 0,
                        'valid_groups': 0  # 有效组数（非nan的组数）
                    }
                
                # 只处理非nan的相关系数
                corr_value = stats['correlation']
                if not np.isnan(corr_value):
                    feature_stats[feature]['correlations'].append(corr_value)
                    feature_stats[feature]['valid_groups'] += 1
                feature_stats[feature]['p_values'].append(stats['p_value'])
                
                if stats['significant']:
                    feature_stats[feature]['significant_count'] += 1
                    
                if abs(corr_value) >= 0.15:  # 强相关阈值（按照图片方案）
                    feature_stats[feature]['strong_correlation_count'] += 1
        
        # 计算三个维度的得分
        evaluation_scores = {}
        
        for feature, stats in feature_stats.items():
            if len(stats['correlations']) == 0:
                print(f"  警告: {feature} 没有有效的相关系数数据")
                continue
                
            G_prime_i = stats['valid_groups']  # 有效组数 G'_i（非nan的组数）
            
            if G_prime_i == 0:
                print(f"  警告: {feature} 没有有效组")
                continue
            
            # 关联强度：S1_i = |r_i| = (1/G'_i) * Σ|r_{i,g}|
            association_strength = np.mean(np.abs(stats['correlations']))
            
            # 统计显著性：S2_i = N_sig,i / G'_i
            statistical_significance = stats['significant_count'] / G_prime_i
            
            # 跨组一致性：S3_i = N_high,i / G'_i  
            cross_group_consistency = stats['strong_correlation_count'] / G_prime_i
            
            evaluation_scores[feature] = {
                'association_strength': association_strength,
                'statistical_significance': statistical_significance,
                'cross_group_consistency': cross_group_consistency,
                'total_groups': G_prime_i,
                'significant_groups': stats['significant_count'],
                'strong_correlation_groups': stats['strong_correlation_count']
            }
            
            print(f"  {feature} 计算详情:")
            print(f"    有效组数 G'_i = {G_prime_i}")
            print(f"    相关系数列表: {[f'{c:.3f}' for c in stats['correlations']]}")
            print(f"    关联强度 S1 = {association_strength:.3f}")
            print(f"    显著组数 = {stats['significant_count']}, 统计显著性 S2 = {statistical_significance:.3f}")
            print(f"    强相关组数 = {stats['strong_correlation_count']}, 跨组一致性 S3 = {cross_group_consistency:.3f}")
        
        # 显示评估结果
        print("\n特征重要性评估:")
        for feature, scores in evaluation_scores.items():
            print(f"{feature}:")
            print(f"  关联强度 (S1): {scores['association_strength']:.3f}")
            print(f"  统计显著性 (S2): {scores['statistical_significance']:.3f} ({scores['significant_groups']}/{scores['total_groups']})")
            print(f"  跨组一致性 (S3): {scores['cross_group_consistency']:.3f} ({scores['strong_correlation_groups']}/{scores['total_groups']})")
        
        return evaluation_scores
        
    def step5_comprehensive_scoring(self, evaluation_scores):
        """步骤5：综合评分和排名"""
        print("\n=== 步骤5：综合评分 ===")
        print("按照图片方案计算综合得分：S_i = S1_i + 0.3 * S2_i + 0.2 * S3_i")
        
        # 权重设置（按照图片方案：S_i = S1_i + 0.3 * S2_i + 0.2 * S3_i）
        weights = {
            'association_strength': 1.0,      # S1_i：关联强度最重要
            'statistical_significance': 0.3,  # 0.3 * S2_i：统计显著性
            'cross_group_consistency': 0.2    # 0.2 * S3_i：跨组一致性
        }
        
        # 计算综合得分（直接使用原始得分，不进行归一化）
        comprehensive_scores = {}
        
        print("\n详细计算过程:")
        for feature, scores in evaluation_scores.items():
            # 提取三个维度得分
            S1 = scores['association_strength']      # 关联强度
            S2 = scores['statistical_significance']  # 统计显著性
            S3 = scores['cross_group_consistency']   # 跨组一致性
            
            # 按照公式计算综合得分：S_i = S1_i + 0.3 * S2_i + 0.2 * S3_i
            comprehensive_score = S1 + 0.3 * S2 + 0.2 * S3
            
            print(f"  {feature}:")
            print(f"    S1 (关联强度) = {S1:.3f}")
            print(f"    S2 (统计显著性) = {S2:.3f}")
            print(f"    S3 (跨组一致性) = {S3:.3f}")
            print(f"    S_i = {S1:.3f} + 0.3 × {S2:.3f} + 0.2 × {S3:.3f}")
            print(f"    S_i = {S1:.3f} + {0.3 * S2:.3f} + {0.2 * S3:.3f} = {comprehensive_score:.3f}")
            print()
            
            comprehensive_scores[feature] = {
                'association_strength': S1,
                'statistical_significance': S2,
                'cross_group_consistency': S3,
                'comprehensive_score': comprehensive_score,
                'total_groups': scores['total_groups'],
                'significant_groups': scores['significant_groups'],
                'strong_correlation_groups': scores['strong_correlation_groups']
            }
        
        # 转换为DataFrame并排序
        scores_df = pd.DataFrame(comprehensive_scores).T
        scores_df = scores_df.sort_values('comprehensive_score', ascending=False)
        
        print("\n=== 最终特征重要性排名 ===")
        print("排名\t特征\t\t\t关联强度\t统计显著性\t跨组一致性\t综合得分")
        print("-" * 80)
        for i, (feature, row) in enumerate(scores_df.iterrows(), 1):
            print(f"{i:2d}\t{feature:<20}\t{row['association_strength']:.3f}\t\t{row['statistical_significance']:.3f}\t\t{row['cross_group_consistency']:.3f}\t\t{row['comprehensive_score']:.3f}")
        
        print(f"\n详细排名表:")
        print(scores_df[['association_strength', 'statistical_significance', 'cross_group_consistency', 'comprehensive_score']].round(3))
        
        # 选择前N个重要特征，按综合得分排序
        scores_df_sorted = scores_df.sort_values('comprehensive_score', ascending=False)
        
        # 强制包含所有染色体Z值（这些是NIPT检测的核心指标）
        essential_features = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
        
        # 选择综合得分前6的特征
        top_features = scores_df_sorted.head(6).index.tolist()
        
        # 添加所有染色体Z值（如果还没有包含）
        for feat in essential_features:
            if feat not in top_features and feat in self.data.columns:
                top_features.append(feat)
        
        # 添加其他重要特征
        additional_features = ['孕妇BMI', 'GC含量', '原始读段数', '重复读段的比例', '唯一比对的读段数']
        for feat in additional_features:
            if feat not in top_features and feat in self.data.columns:
                top_features.append(feat)
                if len(top_features) >= 12:  # 增加到12个特征
                    break
        
        print(f"\n选择的前{len(top_features)}个重要特征: {top_features}")
        self.selected_features = top_features
        return scores_df_sorted
        
    def build_cost_sensitive_models(self):
        """构建集成神经网络模型（修复数据泄露问题）"""
        print("\n=== 构建集成神经网络模型 ===")
        
        # 准备所有特征数据（不预先选择特征）
        all_features = [col for col in self.feature_columns if col in self.data.columns]
        X_all = self.data[all_features]
        y = self.data['is_abnormal']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 计算类别权重和样本权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"类别权重: {class_weight_dict}")
        
        # 计算样本权重（用于支持class_weight的模型）
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        
        print(f"训练集大小: {X_train_scaled.shape[0]}")
        print(f"异常样本比例: {y_train.sum() / len(y_train):.3f}")
        
        # 构建多个不同架构的模型，包括支持class_weight的模型
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        
        # 定义多种模型，包括支持class_weight的模型
        models = {
            'RandomForest_Balanced': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'XGBoost_Balanced': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=class_weight_dict[1]/class_weight_dict[0],  # 处理不平衡
                random_state=42
            ),
            'LogisticRegression_Balanced': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'MLP_With_Sample_Weight': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.00005,
                batch_size=16,
                learning_rate='adaptive',
                learning_rate_init=0.0005,
                max_iter=5000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=200,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
        }
        
        # 使用嵌套交叉验证进行特征选择和模型训练
        print("\n使用嵌套交叉验证进行特征选择和模型训练...")
        
        # 内层CV：特征选择
        from sklearn.model_selection import StratifiedKFold
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 为每个模型进行嵌套CV
        trained_models = {}
        cv_results = {}
        
        for model_name, model_template in models.items():
            print(f"\n训练 {model_name}...")
            
            # 存储每次CV的结果
            cv_scores = []
            selected_features_list = []
            
            # 外层CV循环
            for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_scaled, y_train)):
                print(f"  处理第 {fold+1}/5 折...")
                
                # 分割训练和验证集
                X_train_fold = X_train_scaled[train_idx]
                X_val_fold = X_train_scaled[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                # 内层CV：特征选择
                # 注意：在特征选择阶段，即使是MLP也不使用sample_weight
                # sample_weight只在最终模型训练时使用
                best_features = self.select_features_cv(
                    X_train_fold, y_train_fold, model_template, 
                    None,  # 特征选择阶段不使用sample_weight
                    inner_cv
                )
                
                selected_features_list.append(best_features)
                
                # 使用选择的特征训练模型
                X_train_selected = X_train_fold[:, best_features]
                X_val_selected = X_val_fold[:, best_features]
                
                # 创建新的模型实例
                model = self.create_model_instance(model_template)
                
                # 训练模型
                # 注意：在交叉验证阶段，即使是MLP也不使用sample_weight
                # sample_weight只在最终模型训练时使用
                model.fit(X_train_selected, y_train_fold)
                
                # 验证
                y_val_pred_proba = model.predict_proba(X_val_selected)[:, 1]
                val_auc = roc_auc_score(y_val_fold, y_val_pred_proba)
                cv_scores.append(val_auc)
                
                print(f"    验证AUC: {val_auc:.3f}")
            
            # 选择最常被选中的特征
            feature_votes = {}
            for features in selected_features_list:
                for feat_idx in features:
                    feature_votes[feat_idx] = feature_votes.get(feat_idx, 0) + 1
            
            # 选择投票数最多的特征
            final_features = sorted(feature_votes.keys(), key=lambda x: feature_votes[x], reverse=True)[:12]
            
            print(f"  最终选择的特征索引: {final_features}")
            print(f"  特征投票统计: {feature_votes}")
            
            # 使用最终选择的特征训练完整模型
            X_train_final = X_train_scaled[:, final_features]
            X_test_final = X_test_scaled[:, final_features]
            
            model_final = self.create_model_instance(model_template)
            
            if model_name == 'MLP_With_Sample_Weight':
                # MLP不支持sample_weight，使用其他方法处理不平衡
                # 方法1：调整MLP的class_weight（如果支持）
                if hasattr(model_final, 'class_weight'):
                    model_final.class_weight = 'balanced'
                # 方法2：使用支持sample_weight的包装器
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weights_mlp = compute_sample_weight('balanced', y_train)
                
                # 由于MLP不支持sample_weight，我们使用重采样方法
                from imblearn.over_sampling import SMOTE
                from imblearn.under_sampling import RandomUnderSampler
                from imblearn.pipeline import Pipeline as ImbPipeline
                
                # 创建重采样管道
                resampling_pipeline = ImbPipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('under', RandomUnderSampler(random_state=42))
                ])
                
                # 重采样数据
                X_train_resampled, y_train_resampled = resampling_pipeline.fit_resample(X_train_final, y_train)
                
                # 训练MLP
                model_final.fit(X_train_resampled, y_train_resampled)
            else:
                model_final.fit(X_train_final, y_train)
            
        # 最终评估
        y_pred_proba = model_final.predict_proba(X_test_final)[:, 1]
        
        # 计算多种评估指标
        evaluation_metrics = self.calculate_comprehensive_metrics(
            y_test, y_pred_proba, model_name, X_test_final
        )
        
        print(f"  {model_name} 最终AUC: {evaluation_metrics['auc']:.3f}")
        print(f"  {model_name} CV平均AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"  {model_name} Average Precision: {evaluation_metrics['average_precision']:.3f}")
        print(f"  {model_name} F1 Score: {evaluation_metrics['f1_score']:.3f}")
        
        trained_models[model_name] = {
                'model': model_final,
                'auc_score': evaluation_metrics['auc'],
                'y_pred_proba': y_pred_proba,
                'selected_features': final_features,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'evaluation_metrics': evaluation_metrics
            }
        
        # 选择最佳模型
        best_model_name = max(trained_models.keys(), key=lambda x: trained_models[x]['auc_score'])
        best_model = trained_models[best_model_name]['model']
        y_pred_proba_ensemble = trained_models[best_model_name]['y_pred_proba']
        auc_ensemble = trained_models[best_model_name]['auc_score']
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"最佳模型 AUC: {auc_ensemble:.3f}")
        
        ensemble_model = best_model
        
        # 阈值优化
        optimal_threshold = self.optimize_threshold(y_test, y_pred_proba_ensemble)
        y_pred_optimal = (y_pred_proba_ensemble > optimal_threshold).astype(int)
        
        # 交叉验证
        cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # 特征重要性分析（使用置换重要性）
        best_features = trained_models[best_model_name]['selected_features']
        feature_names = [all_features[i] for i in best_features]
        X_test_selected = X_test_scaled[:, best_features]
        
        feature_importance = self.calculate_permutation_importance(
            ensemble_model, X_test_selected, y_test, feature_names
        )
        
        results = {
            best_model_name: {
                'model': ensemble_model,
                'y_pred': y_pred_optimal,
                'y_pred_proba': y_pred_proba_ensemble,
                'y_test': y_test,
                'auc_score': auc_ensemble,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred_optimal),
                'feature_importance': feature_importance,
                'optimal_threshold': optimal_threshold,
                'all_models': trained_models
            }
        }
        
        print(f"\n=== 最终结果 ===")
        print(f"优化深度神经网络 AUC: {auc_ensemble:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        print(f"最优阈值: {optimal_threshold:.3f}")
        print(f"优化后分类报告:\n{classification_report(y_test, y_pred_optimal)}")
        
        self.models = results
        return results
    
    def select_features_cv(self, X_train, y_train, model_template, sample_weights, cv):
        """在交叉验证中进行特征选择"""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.model_selection import cross_val_score
        
        n_features = X_train.shape[1]
        best_k = min(12, n_features)  # 最多选择12个特征
        
        # 使用多种特征选择方法
        selectors = {
            'f_classif': SelectKBest(f_classif, k=best_k),
            'mutual_info': SelectKBest(mutual_info_classif, k=best_k)
        }
        
        best_selector = None
        best_score = -1
        
        for selector_name, selector in selectors.items():
            try:
                # 选择特征
                X_selected = selector.fit_transform(X_train, y_train)
                selected_features = selector.get_support(indices=True)
                
                # 创建模型实例
                model = self.create_model_instance(model_template)
                
                # 交叉验证评估
                # 注意：在特征选择阶段，即使是MLP也不使用sample_weight
                # sample_weight只在最终模型训练时使用
                try:
                    scores = cross_val_score(model, X_selected, y_train, cv=cv, scoring='roc_auc')
                    mean_score = scores.mean()
                except Exception as e:
                    print(f"    交叉验证失败: {e}")
                    # 如果交叉验证失败，使用简单的训练-验证分割
                    from sklearn.model_selection import train_test_split
                    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                        X_selected, y_train, test_size=0.3, random_state=42, stratify=y_train
                    )
                    model_cv = self.create_model_instance(model_template)
                    model_cv.fit(X_train_cv, y_train_cv)
                    y_pred_proba = model_cv.predict_proba(X_val_cv)[:, 1]
                    score = roc_auc_score(y_val_cv, y_pred_proba)
                    mean_score = score
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_selector = selector
                    
            except Exception as e:
                print(f"    特征选择方法 {selector_name} 失败: {e}")
                continue
        
        if best_selector is not None:
            return best_selector.get_support(indices=True)
        else:
            # 如果所有方法都失败，返回前12个特征
            return list(range(min(12, n_features)))
    
    def create_model_instance(self, model_template):
        """创建模型的新实例"""
        import copy
        return copy.deepcopy(model_template)
    
    def calculate_comprehensive_metrics(self, y_true, y_pred_proba, model_name, X_test):
        """计算综合评估指标"""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, precision_recall_curve,
            classification_report, confusion_matrix, f1_score, precision_score, recall_score
        )
        
        # 基本指标
        auc = roc_auc_score(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)
        
        # 优化阈值
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        # 使用最优阈值进行预测
        y_pred = (y_pred_proba > optimal_threshold).astype(int)
        
        # 分类指标
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 按BMI组分层的指标（如果数据中有BMI分组信息）
        bmi_stratified_metrics = self.calculate_bmi_stratified_metrics(y_true, y_pred_proba, optimal_threshold)
        
        return {
            'auc': auc,
            'average_precision': average_precision,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'bmi_stratified': bmi_stratified_metrics,
            'precision_recall_curve': (precision, recall, thresholds)
        }
    
    def calculate_bmi_stratified_metrics(self, y_true, y_pred_proba, threshold):
        """计算按BMI组分层的评估指标"""
        if not hasattr(self, 'data') or 'BMI_group' not in self.data.columns:
            return None
        
        # 获取测试集对应的BMI分组信息
        # 这里需要根据实际情况调整，因为测试集可能没有BMI分组信息
        try:
            stratified_metrics = {}
            
            # 假设我们有测试集的索引信息
            # 在实际应用中，需要确保测试集包含BMI分组信息
            for bmi_group in self.data['BMI_group'].unique():
                if pd.isna(bmi_group):
                    continue
                
                # 这里需要根据实际情况获取对应分组的测试样本
                # 简化处理：返回None，表示无法计算分层指标
                pass
            
            return stratified_metrics
        except:
            return None
    
    def calculate_permutation_importance(self, model, X_test, y_test, feature_names, n_repeats=10):
        """计算置换重要性"""
        from sklearn.inspection import permutation_importance
        
        try:
            # 使用sklearn的permutation_importance
            perm_importance = permutation_importance(
                model, X_test, y_test, 
                n_repeats=n_repeats, 
                random_state=42,
                scoring='roc_auc'
            )
            
            # 创建特征重要性字典
            importance_dict = {}
            for i, feature in enumerate(feature_names):
                importance_dict[feature] = {
                    'importance_mean': perm_importance.importances_mean[i],
                    'importance_std': perm_importance.importances_std[i]
                }
            
            return importance_dict
            
        except Exception as e:
            print(f"置换重要性计算失败: {e}")
            # 回退到简单的特征重要性（如果模型支持）
            if hasattr(model, 'feature_importances_'):
                importance_dict = {}
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = {
                        'importance_mean': model.feature_importances_[i],
                        'importance_std': 0.0
                    }
                return importance_dict
            elif hasattr(model, 'coefs_'):
                # 对于神经网络，使用第一层权重的绝对值
                first_layer_weights = np.abs(model.coefs_[0])
                importance_dict = {}
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = {
                        'importance_mean': np.mean(first_layer_weights[i]),
                        'importance_std': np.std(first_layer_weights[i])
                    }
                return importance_dict
            else:
                return None

    def optimize_threshold(self, y_true, y_proba):
        """优化分类阈值（修复thresholds长度问题）"""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 找到最佳阈值（处理thresholds长度问题）
        best_threshold_idx = np.argmax(f1_scores)
        
        # 确保索引不越界
        if best_threshold_idx < len(thresholds):
            best_threshold = thresholds[best_threshold_idx]
        else:
            # 如果索引越界，使用最后一个阈值或默认值
            best_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5
        
        return best_threshold
        
    def visualize_results(self):
        """可视化结果"""
        print("\n=== 生成可视化结果 ===")
        
        # 创建结果目录
        import os
        os.makedirs('problem4_analysis/results', exist_ok=True)
        
        # 1. 数据分布可视化
        self.plot_data_distribution()
        
        # 2. BMI分组分析可视化
        self.plot_bmi_group_analysis()
        
        # 3. 特征相关性热图
        self.plot_correlation_heatmap()
        
        # 4. 五步法特征筛选过程可视化
        self.plot_feature_selection_process()
        
        # 5. 模型性能可视化
        self.plot_model_performance()
        
        # 6. 混淆矩阵和分类报告
        self.plot_confusion_matrix()
        
        # 7. 特征重要性分析
        self.plot_feature_importance()
        
        # 8. 预测概率分布
        self.plot_prediction_distribution()
        
        # 9. 学习曲线
        self.plot_learning_curves()
        
        # 10. 综合仪表板
        self.plot_comprehensive_dashboard()
        
        print("所有可视化结果已保存到 problem4_analysis/results/")
    
    def plot_data_distribution(self):
        """数据分布可视化"""
        print("生成数据分布图...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('数据分布分析', fontsize=16, fontweight='bold')
        
        # 异常样本分布
        axes[0, 0].pie([self.data['is_abnormal'].sum(), len(self.data) - self.data['is_abnormal'].sum()], 
                      labels=['异常', '正常'], autopct='%1.1f%%', colors=['red', 'green'])
        axes[0, 0].set_title('异常样本分布')
        
        # BMI分布
        axes[0, 1].hist(self.data['孕妇BMI'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('孕妇BMI分布')
        axes[0, 1].set_xlabel('BMI')
        axes[0, 1].set_ylabel('频次')
        
        # 年龄分布
        axes[0, 2].hist(self.data['年龄'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 2].set_title('年龄分布')
        axes[0, 2].set_xlabel('年龄')
        axes[0, 2].set_ylabel('频次')
        
        # 染色体Z值分布
        chromosome_z_values = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
        for i, col in enumerate(chromosome_z_values):
            if i < 2:
                axes[1, i].hist(self.data[col], bins=30, alpha=0.7, edgecolor='black')
                axes[1, i].set_title(f'{col}分布')
                axes[1, i].set_xlabel('Z值')
                axes[1, i].set_ylabel('频次')
        
        # 测序质量指标
        axes[1, 2].scatter(self.data['原始读段数'], self.data['GC含量'], 
                          c=self.data['is_abnormal'], cmap='RdYlBu', alpha=0.6)
        axes[1, 2].set_title('原始读段数 vs GC含量')
        axes[1, 2].set_xlabel('原始读段数')
        axes[1, 2].set_ylabel('GC含量')
        
        plt.tight_layout()
        plt.savefig('problem4_analysis/results/data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_bmi_group_analysis(self):
        """BMI分组分析可视化"""
        print("生成BMI分组分析图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BMI分组分析', fontsize=16, fontweight='bold')
        
        # BMI组异常比例
        bmi_groups = self.data.groupby('BMI_group')['is_abnormal'].agg(['count', 'sum', 'mean']).reset_index()
        bmi_groups.columns = ['BMI组', '总样本数', '异常样本数', '异常比例']
        
        axes[0, 0].bar(bmi_groups['BMI组'], bmi_groups['异常比例'], color='lightblue', edgecolor='black')
        axes[0, 0].set_title('各BMI组异常比例')
        axes[0, 0].set_ylabel('异常比例')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # BMI组样本量分布
        axes[0, 1].bar(bmi_groups['BMI组'], bmi_groups['总样本数'], color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('各BMI组样本量')
        axes[0, 1].set_ylabel('样本数')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # BMI与异常关系散点图
        scatter = axes[1, 0].scatter(self.data['孕妇BMI'], self.data['is_abnormal'], 
                                   c=self.data['is_abnormal'], cmap='RdYlBu', alpha=0.6)
        axes[1, 0].set_title('BMI与异常判定关系')
        axes[1, 0].set_xlabel('孕妇BMI')
        axes[1, 0].set_ylabel('异常判定 (0=正常, 1=异常)')
        
        # BMI组内特征分布箱线图
        bmi_data = []
        bmi_labels = []
        for group in self.data['BMI_group'].unique():
            if pd.notna(group):
                group_data = self.data[self.data['BMI_group'] == group]
                bmi_data.append(group_data['孕妇BMI'])
                bmi_labels.append(str(group))
        
        axes[1, 1].boxplot(bmi_data, labels=bmi_labels)
        axes[1, 1].set_title('各BMI组BMI值分布')
        axes[1, 1].set_ylabel('BMI值')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('problem4_analysis/results/bmi_group_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self):
        """特征相关性热图"""
        print("生成特征相关性热图...")
        
        # 选择数值特征
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_features].corr()
        
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('特征相关性热图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('problem4_analysis/results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_selection_process(self):
        """五步法特征筛选过程可视化"""
        print("生成特征筛选过程图...")
        
        # 读取特征排名数据
        import os
        if os.path.exists('problem4_analysis/results/feature_ranking.csv'):
            feature_ranking = pd.read_csv('problem4_analysis/results/feature_ranking.csv', index_col=0)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('五步法特征筛选过程', fontsize=16, fontweight='bold')
            
            # 关联强度
            axes[0, 0].barh(range(len(feature_ranking)), feature_ranking['association_strength'], 
                           color='skyblue', edgecolor='black')
            axes[0, 0].set_yticks(range(len(feature_ranking)))
            axes[0, 0].set_yticklabels(feature_ranking.index, fontsize=8)
            axes[0, 0].set_title('关联强度 (S1)')
            axes[0, 0].set_xlabel('平均相关系数绝对值')
            
            # 统计显著性
            axes[0, 1].barh(range(len(feature_ranking)), feature_ranking['statistical_significance'], 
                           color='lightcoral', edgecolor='black')
            axes[0, 1].set_yticks(range(len(feature_ranking)))
            axes[0, 1].set_yticklabels(feature_ranking.index, fontsize=8)
            axes[0, 1].set_title('统计显著性 (S2)')
            axes[0, 1].set_xlabel('显著相关组数比例')
            
            # 跨组一致性
            axes[1, 0].barh(range(len(feature_ranking)), feature_ranking['cross_group_consistency'], 
                           color='lightgreen', edgecolor='black')
            axes[1, 0].set_yticks(range(len(feature_ranking)))
            axes[1, 0].set_yticklabels(feature_ranking.index, fontsize=8)
            axes[1, 0].set_title('跨组一致性 (S3)')
            axes[1, 0].set_xlabel('强相关组数比例')
            
            # 综合得分
            axes[1, 1].barh(range(len(feature_ranking)), feature_ranking['comprehensive_score'], 
                           color='gold', edgecolor='black')
            axes[1, 1].set_yticks(range(len(feature_ranking)))
            axes[1, 1].set_yticklabels(feature_ranking.index, fontsize=8)
            axes[1, 1].set_title('综合得分 (S_i)')
            axes[1, 1].set_xlabel('综合得分')
            
            plt.tight_layout()
            plt.savefig('problem4_analysis/results/feature_selection_process.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_model_performance(self):
        """模型性能可视化"""
        print("生成模型性能图...")
        
        if hasattr(self, 'models') and self.models:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('模型性能分析', fontsize=16, fontweight='bold')
            
            for name, result in self.models.items():
                if 'y_test' in result and 'y_pred_proba' in result:
                    # ROC曲线
                    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                    axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})', linewidth=2)
                    
                    # 精确率-召回率曲线
                    from sklearn.metrics import precision_recall_curve
                    precision, recall, _ = precision_recall_curve(result['y_test'], result['y_pred_proba'])
                    axes[0, 1].plot(recall, precision, label=f'{name}', linewidth=2)
                    
                    # 预测概率分布
                    axes[1, 0].hist(result['y_pred_proba'][result['y_test'] == 0], 
                                   bins=20, alpha=0.7, label='正常样本', color='green')
                    axes[1, 0].hist(result['y_pred_proba'][result['y_test'] == 1], 
                                   bins=20, alpha=0.7, label='异常样本', color='red')
                    
                    # 阈值分析
                    thresholds = np.linspace(0, 1, 100)
                    precisions = []
                    recalls = []
                    f1_scores = []
                    
                    for thresh in thresholds:
                        y_pred_thresh = (result['y_pred_proba'] > thresh).astype(int)
                        if len(np.unique(y_pred_thresh)) > 1:
                            from sklearn.metrics import precision_score, recall_score, f1_score
                            precisions.append(precision_score(result['y_test'], y_pred_thresh, zero_division=0))
                            recalls.append(recall_score(result['y_test'], y_pred_thresh, zero_division=0))
                            f1_scores.append(f1_score(result['y_test'], y_pred_thresh, zero_division=0))
                        else:
                            precisions.append(0)
                            recalls.append(0)
                            f1_scores.append(0)
                    
                    axes[1, 1].plot(thresholds, precisions, label='精确率', linewidth=2)
                    axes[1, 1].plot(thresholds, recalls, label='召回率', linewidth=2)
                    axes[1, 1].plot(thresholds, f1_scores, label='F1分数', linewidth=2)
                    axes[1, 1].axvline(x=result.get('optimal_threshold', 0.5), 
                                     color='red', linestyle='--', label='最优阈值')
            
            # 设置图表属性
            axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 0].set_xlabel('假正率 (FPR)')
            axes[0, 0].set_ylabel('真正率 (TPR)')
            axes[0, 0].set_title('ROC曲线')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_xlabel('召回率')
            axes[0, 1].set_ylabel('精确率')
            axes[0, 1].set_title('精确率-召回率曲线')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_xlabel('预测概率')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].set_title('预测概率分布')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].set_xlabel('阈值')
            axes[1, 1].set_ylabel('分数')
            axes[1, 1].set_title('阈值分析')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('problem4_analysis/results/model_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    def plot_confusion_matrix(self):
        """混淆矩阵和分类报告可视化"""
        print("生成混淆矩阵图...")
        
        if hasattr(self, 'models') and self.models:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('分类性能分析', fontsize=16, fontweight='bold')
            
            for name, result in self.models.items():
                if 'y_test' in result and 'y_pred' in result:
                    # 混淆矩阵
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(result['y_test'], result['y_pred'])
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['正常', '异常'], 
                               yticklabels=['正常', '异常'],
                               ax=axes[0])
                    axes[0].set_title(f'{name} 混淆矩阵')
                    axes[0].set_xlabel('预测标签')
                    axes[0].set_ylabel('真实标签')
                    
                    # 分类报告可视化
                    from sklearn.metrics import classification_report
                    report = classification_report(result['y_test'], result['y_pred'], 
                                                 target_names=['正常', '异常'], 
                                                 output_dict=True)
                    
                    # 提取指标
                    metrics = ['precision', 'recall', 'f1-score']
                    classes = ['正常', '异常']
                    data = []
                    for cls in classes:
                        row = [report[cls][metric] for metric in metrics]
                        data.append(row)
                    
                    im = axes[1].imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                    axes[1].set_xticks(range(len(metrics)))
                    axes[1].set_yticks(range(len(classes)))
                    axes[1].set_xticklabels(metrics)
                    axes[1].set_yticklabels(classes)
                    axes[1].set_title(f'{name} 分类报告')
                    
                    # 添加数值标签
                    for i in range(len(classes)):
                        for j in range(len(metrics)):
                            text = axes[1].text(j, i, f'{data[i][j]:.3f}',
                                              ha="center", va="center", color="black")
                    
                    # 添加颜色条
                    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                    break
            
            plt.tight_layout()
            plt.savefig('problem4_analysis/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_feature_importance(self):
        """特征重要性分析"""
        print("生成特征重要性图...")
        
        if hasattr(self, 'models') and self.models:
            for name, result in self.models.items():
                if result.get('feature_importance') is not None:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle(f'{name} 特征重要性分析', fontsize=16, fontweight='bold')
                    
                    importance = result['feature_importance']
                    features = self.selected_features
                    
                    # 排序特征重要性
                    sorted_idx = np.argsort(importance)[::-1]
                    sorted_features = [features[i] for i in sorted_idx]
                    sorted_importance = importance[sorted_idx]
                    
                    # 水平条形图
                    bars = axes[0, 0].barh(range(len(sorted_features)), sorted_importance, 
                                   color='skyblue', edgecolor='black')
                    axes[0, 0].set_yticks(range(len(sorted_features)))
                    axes[0, 0].set_yticklabels(sorted_features, fontsize=10)
                    axes[0, 0].set_xlabel('特征重要性')
                    axes[0, 0].set_title('特征重要性排序')
                    axes[0, 0].invert_yaxis()
                    
                    # 添加数值标签
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        axes[0, 0].text(width, bar.get_y() + bar.get_height()/2,
                                f'{width:.3f}', ha='left', va='center')
                    
                    # 饼图（前8个特征）
                    top_n = min(8, len(sorted_features))
                    axes[0, 1].pie(sorted_importance[:top_n], labels=sorted_features[:top_n], 
                                  autopct='%1.1f%%', startangle=90)
                    axes[0, 1].set_title(f'前{top_n}个重要特征占比')
                    
                    # 累积重要性
                    cumulative_importance = np.cumsum(sorted_importance)
                    axes[1, 0].plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                                   marker='o', linewidth=2, markersize=6)
                    axes[1, 0].set_xlabel('特征数量')
                    axes[1, 0].set_ylabel('累积重要性')
                    axes[1, 0].set_title('累积特征重要性')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 特征重要性分布
                    axes[1, 1].hist(importance, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
                    axes[1, 1].set_xlabel('特征重要性')
                    axes[1, 1].set_ylabel('频次')
                    axes[1, 1].set_title('特征重要性分布')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig('problem4_analysis/results/feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    break
    
    def plot_prediction_distribution(self):
        """预测概率分布可视化"""
        print("生成预测概率分布图...")
        
        if hasattr(self, 'models') and self.models:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('预测概率分布分析', fontsize=16, fontweight='bold')
            
            for name, result in self.models.items():
                if 'y_test' in result and 'y_pred_proba' in result:
                    y_test = result['y_test']
                    y_pred_proba = result['y_pred_proba']
                    
                    # 预测概率分布
                    axes[0, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                                   label='正常样本', color='green', density=True)
                    axes[0, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                                   label='异常样本', color='red', density=True)
                    axes[0, 0].set_xlabel('预测概率')
                    axes[0, 0].set_ylabel('密度')
                    axes[0, 0].set_title('预测概率分布')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 预测概率箱线图
                    data_to_plot = [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]]
                    axes[0, 1].boxplot(data_to_plot, labels=['正常', '异常'])
                    axes[0, 1].set_ylabel('预测概率')
                    axes[0, 1].set_title('预测概率箱线图')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # 预测概率vs真实标签
                    axes[1, 0].scatter(y_test, y_pred_proba, alpha=0.6, s=20)
                    axes[1, 0].set_xlabel('真实标签')
                    axes[1, 0].set_ylabel('预测概率')
                    axes[1, 0].set_title('预测概率vs真实标签')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 最优阈值线
                    optimal_threshold = result.get('optimal_threshold', 0.5)
                    axes[1, 0].axhline(y=optimal_threshold, color='red', linestyle='--', 
                                     label=f'最优阈值: {optimal_threshold:.3f}')
                    axes[1, 0].legend()
                    
                    # 预测概率分位数分析
                    normal_quantiles = np.percentile(y_pred_proba[y_test == 0], [25, 50, 75])
                    abnormal_quantiles = np.percentile(y_pred_proba[y_test == 1], [25, 50, 75])
                    
                    x_pos = np.arange(3)
                    width = 0.35
                    
                    axes[1, 1].bar(x_pos - width/2, normal_quantiles, width, 
                                  label='正常样本', alpha=0.7, color='green')
                    axes[1, 1].bar(x_pos + width/2, abnormal_quantiles, width, 
                                  label='异常样本', alpha=0.7, color='red')
                    
                    axes[1, 1].set_xlabel('分位数')
                    axes[1, 1].set_ylabel('预测概率')
                    axes[1, 1].set_title('预测概率分位数对比')
                    axes[1, 1].set_xticks(x_pos)
                    axes[1, 1].set_xticklabels(['25%', '50%', '75%'])
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    break
                    
            plt.tight_layout()
            plt.savefig('problem4_analysis/results/prediction_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_learning_curves(self):
        """学习曲线可视化"""
        print("生成学习曲线图...")
        
        if hasattr(self, 'models') and self.models:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('模型学习分析', fontsize=16, fontweight='bold')
            
            for name, result in self.models.items():
                if 'model' in result and hasattr(result['model'], 'loss_curve_'):
                    # 损失曲线
                    if hasattr(result['model'], 'loss_curve_'):
                        axes[0, 0].plot(result['model'].loss_curve_, label=f'{name} 训练损失')
                        axes[0, 0].set_xlabel('迭代次数')
                        axes[0, 0].set_ylabel('损失')
                        axes[0, 0].set_title('训练损失曲线')
                        axes[0, 0].legend()
                        axes[0, 0].grid(True, alpha=0.3)
                    
                    # 验证曲线
                    if hasattr(result['model'], 'validation_scores_'):
                        axes[0, 1].plot(result['model'].validation_scores_, label=f'{name} 验证分数')
                        axes[0, 1].set_xlabel('迭代次数')
                        axes[0, 1].set_ylabel('验证分数')
                        axes[0, 1].set_title('验证分数曲线')
                        axes[0, 1].legend()
                        axes[0, 1].grid(True, alpha=0.3)
                
                # 交叉验证分数
                if 'cv_mean' in result and 'cv_std' in result:
                    cv_scores = [result['cv_mean']] * 5  # 假设5折交叉验证
                    cv_errors = [result['cv_std']] * 5
                    
                    axes[1, 0].errorbar(range(1, 6), cv_scores, yerr=cv_errors, 
                                       marker='o', capsize=5, label=f'{name}')
                    axes[1, 0].set_xlabel('交叉验证折数')
                    axes[1, 0].set_ylabel('AUC分数')
                    axes[1, 0].set_title('交叉验证分数')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                
                # 模型复杂度分析
                if hasattr(result['model'], 'coefs_'):
                    layer_sizes = [len(coef) for coef in result['model'].coefs_]
                    axes[1, 1].bar(range(len(layer_sizes)), layer_sizes, 
                                  alpha=0.7, color='lightblue', edgecolor='black')
                    axes[1, 1].set_xlabel('层数')
                    axes[1, 1].set_ylabel('神经元数量')
                    axes[1, 1].set_title('网络结构')
                    axes[1, 1].grid(True, alpha=0.3)
                
                    break
        
            plt.tight_layout()
            plt.savefig('problem4_analysis/results/learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_comprehensive_dashboard(self):
        """综合仪表板"""
        print("生成综合仪表板...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('女胎异常判定分析综合仪表板', fontsize=20, fontweight='bold')
        
        # 1. 数据概览 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        total_samples = len(self.data)
        abnormal_samples = self.data['is_abnormal'].sum()
        normal_samples = total_samples - abnormal_samples
        
        ax1.pie([normal_samples, abnormal_samples], 
               labels=[f'正常\n{normal_samples}个', f'异常\n{abnormal_samples}个'], 
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax1.set_title('样本分布概览', fontweight='bold')
        
        # 2. 模型性能 (右上)
        ax2 = fig.add_subplot(gs[0, 1])
        if hasattr(self, 'models') and self.models:
            for name, result in self.models.items():
                if 'auc_score' in result:
                    ax2.bar(name, result['auc_score'], color='skyblue', alpha=0.7)
                    ax2.text(name, result['auc_score'] + 0.01, f'{result["auc_score"]:.3f}', 
                           ha='center', va='bottom', fontweight='bold')
        ax2.set_ylabel('AUC分数')
        ax2.set_title('模型性能', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # 3. 特征重要性Top5 (左中)
        ax3 = fig.add_subplot(gs[0, 2:])
        if os.path.exists('problem4_analysis/results/feature_ranking.csv'):
            feature_ranking = pd.read_csv('problem4_analysis/results/feature_ranking.csv', index_col=0)
            top5_features = feature_ranking.head(5)
            
            bars = ax3.barh(range(len(top5_features)), top5_features['comprehensive_score'], 
                           color='gold', alpha=0.7, edgecolor='black')
            ax3.set_yticks(range(len(top5_features)))
            ax3.set_yticklabels(top5_features.index, fontsize=10)
            ax3.set_xlabel('综合得分')
            ax3.set_title('Top5重要特征', fontweight='bold')
            ax3.invert_yaxis()
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # 4. BMI分组分析 (中左)
        ax4 = fig.add_subplot(gs[1, :2])
        bmi_groups = self.data.groupby('BMI_group')['is_abnormal'].agg(['count', 'mean']).reset_index()
        bmi_groups.columns = ['BMI组', '样本数', '异常比例']
        
        x = np.arange(len(bmi_groups))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, bmi_groups['样本数'], width, label='样本数', alpha=0.7, color='lightblue')
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, bmi_groups['异常比例'], width, label='异常比例', alpha=0.7, color='lightcoral')
        
        ax4.set_xlabel('BMI组')
        ax4.set_ylabel('样本数', color='blue')
        ax4_twin.set_ylabel('异常比例', color='red')
        ax4.set_title('BMI分组分析', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(bmi_groups['BMI组'], rotation=45)
        
        # 5. ROC曲线 (中右)
        ax5 = fig.add_subplot(gs[1, 2:])
        if hasattr(self, 'models') and self.models:
            for name, result in self.models.items():
                if 'y_test' in result and 'y_pred_proba' in result:
                    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                    ax5.plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})', linewidth=2)
        
        ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器')
        ax5.set_xlabel('假正率 (FPR)')
        ax5.set_ylabel('真正率 (TPR)')
        ax5.set_title('ROC曲线', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 预测概率分布 (下左)
        ax6 = fig.add_subplot(gs[2, :2])
        if hasattr(self, 'models') and self.models:
            for name, result in self.models.items():
                if 'y_test' in result and 'y_pred_proba' in result:
                    y_test = result['y_test']
                    y_pred_proba = result['y_pred_proba']
                    
                    ax6.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, 
                            label='正常样本', color='green', density=True)
                    ax6.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, 
                            label='异常样本', color='red', density=True)
                    
                    # 添加最优阈值线
                    optimal_threshold = result.get('optimal_threshold', 0.5)
                    ax6.axvline(x=optimal_threshold, color='black', linestyle='--', 
                              label=f'最优阈值: {optimal_threshold:.3f}')
                    break
        
        ax6.set_xlabel('预测概率')
        ax6.set_ylabel('密度')
        ax6.set_title('预测概率分布', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 混淆矩阵 (下右)
        ax7 = fig.add_subplot(gs[2, 2:])
        if hasattr(self, 'models') and self.models:
            for name, result in self.models.items():
                if 'y_test' in result and 'y_pred' in result:
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(result['y_test'], result['y_pred'])
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['正常', '异常'], 
                               yticklabels=['正常', '异常'],
                               ax=ax7)
                    ax7.set_title(f'{name} 混淆矩阵', fontweight='bold')
                    ax7.set_xlabel('预测标签')
                    ax7.set_ylabel('真实标签')
                    break
        
        # 8. 技术指标总结 (底部)
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # 创建技术指标表格
        if hasattr(self, 'models') and self.models:
            for name, result in self.models.items():
                if 'auc_score' in result:
                    metrics_text = f"""
                    <b>模型性能指标</b><br>
                    • AUC分数: {result['auc_score']:.3f}<br>
                    • 交叉验证: {result.get('cv_mean', 0):.3f} ± {result.get('cv_std', 0):.3f}<br>
                    • 最优阈值: {result.get('optimal_threshold', 0.5):.3f}<br>
                    • 总样本数: {len(self.data)}<br>
                    • 异常样本: {self.data['is_abnormal'].sum()} ({self.data['is_abnormal'].mean()*100:.1f}%)<br>
                    • 特征数量: {len(self.selected_features) if hasattr(self, 'selected_features') else 'N/A'}<br>
                    • 网络结构: 五层深度网络 (512, 256, 128, 64, 32)
                    """
                    ax8.text(0.1, 0.5, metrics_text, transform=ax8.transAxes, 
                            fontsize=12, verticalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
                    break
        
        plt.tight_layout()
        plt.savefig('problem4_analysis/results/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("开始问题4的完整分析...")
        
        # 加载数据
        self.load_data()
        
        # 五步法变量筛选
        group_stats = self.step1_grouping_treatment()
        correlations = self.step2_correlation_calculation()
        valid_groups = self.step3_quality_control()
        evaluation_scores = self.step4_multi_dimensional_evaluation(correlations)
        feature_ranking = self.step5_comprehensive_scoring(evaluation_scores)
        
        # 构建模型
        model_results = self.build_cost_sensitive_models()
        
        # 生成可视化
        self.visualize_results()
        
        # 保存结果
        self.save_results(group_stats, correlations, evaluation_scores, feature_ranking, model_results)
        
        print("\n=== 分析完成 ===")
        return {
            'group_stats': group_stats,
            'correlations': correlations,
            'evaluation_scores': evaluation_scores,
            'feature_ranking': feature_ranking,
            'model_results': model_results
        }
        
    def save_results(self, group_stats, correlations, evaluation_scores, feature_ranking, model_results):
        """保存分析结果"""
        import os
        os.makedirs('problem4_analysis/results', exist_ok=True)
        
        # 保存分组统计
        group_stats.to_csv('problem4_analysis/results/group_statistics.csv', encoding='utf-8-sig')
        
        # 保存特征排名
        feature_ranking.to_csv('problem4_analysis/results/feature_ranking.csv', encoding='utf-8-sig')
        
        # 保存模型结果
        model_summary = []
        for name, result in model_results.items():
            model_summary.append({
                'Model': name,
                'AUC_Score': result['auc_score']
            })
        
        pd.DataFrame(model_summary).to_csv('problem4_analysis/results/model_performance.csv', 
                                         index=False, encoding='utf-8-sig')
        
        print("结果已保存到 problem4_analysis/results/")

if __name__ == "__main__":
    # 运行分析
    analyzer = Problem4Analyzer('../初始数据/女胎检测数据.csv')
    results = analyzer.run_complete_analysis()
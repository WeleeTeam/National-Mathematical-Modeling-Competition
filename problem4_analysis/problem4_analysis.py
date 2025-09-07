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
                    # 使用scipy计算（用于验证）
                    from scipy import stats as scipy_stats
                    corr_scipy, p_value = scipy_stats.spearmanr(group_data[feature], group_data['is_abnormal'])
                    
                    # 手动计算Spearman相关系数（为第一个特征显示详细过程）
                    show_details = (i == 0)  # 只为第一个特征显示详细计算过程
                    corr_manual = self.calculate_spearman_manual(
                        group_data[feature], group_data['is_abnormal'], show_details=show_details
                    )
                    
                    group_correlations[feature] = {
                        'correlation': corr_scipy,
                        'correlation_manual': corr_manual,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            correlations[group] = group_correlations
            
            # 显示详细的计算结果
            print(f"\n详细计算结果:")
            for feature, stats in group_correlations.items():
                print(f"  {feature}:")
                
                # 处理nan值
                scipy_result = stats['correlation']
                manual_result = stats['correlation_manual']
                p_value = stats['p_value']
                
                if np.isnan(scipy_result):
                    print(f"    Scipy结果: nan (常数变量)")
                else:
                    print(f"    Scipy结果: {scipy_result:.6f}")
                
                if np.isnan(manual_result):
                    print(f"    手动计算: nan (常数变量)")
                else:
                    print(f"    手动计算: {manual_result:.6f}")
                
                if not np.isnan(scipy_result) and not np.isnan(manual_result):
                    print(f"    差异: {abs(scipy_result - manual_result):.8f}")
                else:
                    print(f"    差异: 无法比较 (存在nan值)")
                
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
        
    def calculate_spearman_manual(self, x, y, show_details=False):
        """手动计算Spearman相关系数"""
        from scipy import stats as scipy_stats
        
        # 检查是否为常数变量
        if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
            if show_details:
                print(f"    警告: 检测到常数变量，无法计算相关系数")
                print(f"      x的唯一值数量: {len(np.unique(x))}")
                print(f"      y的唯一值数量: {len(np.unique(y))}")
            return np.nan
        
        # 计算秩次
        x_ranks = scipy_stats.rankdata(x)
        y_ranks = scipy_stats.rankdata(y)
        
        # 计算秩次差
        d = x_ranks - y_ranks
        
        # 计算秩次差的平方和
        d_squared_sum = np.sum(d ** 2)
        
        # 样本量
        n = len(x)
        
        # Spearman相关系数公式：r = 1 - (6 * Σd²) / (n * (n² - 1))
        if n > 1:
            r = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        else:
            r = np.nan
        
        if show_details:
            print(f"    详细计算过程:")
            print(f"      样本量 n = {n}")
            print(f"      x的唯一值数量: {len(np.unique(x))}")
            print(f"      y的唯一值数量: {len(np.unique(y))}")
            print(f"      Σd² = {d_squared_sum}")
            print(f"      n(n²-1) = {n * (n**2 - 1)}")
            print(f"      6 * Σd² = {6 * d_squared_sum}")
            print(f"      6 * Σd² / (n * (n² - 1)) = {(6 * d_squared_sum) / (n * (n**2 - 1)):.6f}")
            print(f"      r = 1 - {(6 * d_squared_sum) / (n * (n**2 - 1)):.6f} = {r:.6f}")
            
        return r
        
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
        """构建集成神经网络模型"""
        print("\n=== 构建集成神经网络模型 ===")
        
        # 准备数据
        X = self.data[self.selected_features]
        y = self.data['is_abnormal']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"类别权重: {class_weight_dict}")
        
        print(f"训练集大小: {X_train_scaled.shape[0]}")
        print(f"异常样本比例: {y_train.sum() / len(y_train):.3f}")
        
        # 构建多个不同架构的神经网络
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import VotingClassifier
        
        # 定义优化的单神经网络模型（基于效果最好的NN_Deep进行优化）
        neural_networks = {
            'NN_Optimized_Deep': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64, 32),  # 五层更深的网络
            activation='relu',
            solver='adam',
                alpha=0.00005,  # 减少正则化，允许更复杂的学习
                batch_size=16,  # 减小批次大小，提高学习稳定性
            learning_rate='adaptive',
                learning_rate_init=0.0005,  # 降低初始学习率
                max_iter=5000,  # 增加最大迭代次数
            random_state=42,
            early_stopping=True,
                validation_fraction=0.15,  # 减少验证集比例，增加训练数据
                n_iter_no_change=200,  # 增加耐心值
                beta_1=0.9,  # Adam优化器参数
                beta_2=0.999,
                epsilon=1e-8
            )
        }
        
        # 训练优化的单神经网络模型
        print("\n训练优化的深度神经网络模型...")
        model_name = list(neural_networks.keys())[0]
        model = neural_networks[model_name]
        
        print(f"模型架构: {model_name}")
        print(f"网络结构: {model.hidden_layer_sizes}")
        print(f"激活函数: {model.activation}")
        print(f"优化器: {model.solver}")
        print(f"正则化参数: {model.alpha}")
        print(f"批次大小: {model.batch_size}")
        print(f"最大迭代次数: {model.max_iter}")
        
        # 训练模型
        print("\n开始训练...")
        model.fit(X_train_scaled, y_train)
        
        # 预测和评估
        y_pred_proba_ensemble = model.predict_proba(X_test_scaled)[:, 1]
        auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
        
        print(f"\n训练完成!")
        print(f"模型 AUC: {auc_ensemble:.3f}")
        
        # 检查是否收敛
        if hasattr(model, 'n_iter_'):
            print(f"实际迭代次数: {model.n_iter_}")
            if model.n_iter_ == model.max_iter:
                print("警告: 模型可能未完全收敛，考虑增加max_iter")
            else:
                print("模型已收敛")
        
        ensemble_model = model
        
        # 阈值优化
        optimal_threshold = self.optimize_threshold(y_test, y_pred_proba_ensemble)
        y_pred_optimal = (y_pred_proba_ensemble > optimal_threshold).astype(int)
        
        # 交叉验证
        cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # 特征重要性分析
        feature_importance = None
        if hasattr(ensemble_model, 'coefs_'):
            first_layer_weights = np.abs(ensemble_model.coefs_[0])
            feature_importance = np.mean(first_layer_weights, axis=1)
        
        results = {
            'Optimized Deep Neural Network': {
                'model': ensemble_model,
                'y_pred': y_pred_optimal,
                'y_pred_proba': y_pred_proba_ensemble,
                'y_test': y_test,
                'auc_score': auc_ensemble,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred_optimal),
                'feature_importance': feature_importance,
                'optimal_threshold': optimal_threshold
            }
        }
        
        print(f"\n=== 最终结果 ===")
        print(f"优化深度神经网络 AUC: {auc_ensemble:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        print(f"最优阈值: {optimal_threshold:.3f}")
        print(f"优化后分类报告:\n{classification_report(y_test, y_pred_optimal)}")
        
        self.models = results
        return results
    
    def optimize_threshold(self, y_true, y_proba):
        """优化分类阈值"""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 找到最佳阈值
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
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
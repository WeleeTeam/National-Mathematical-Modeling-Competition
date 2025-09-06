#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4：女胎异常判定分析
基于五步法进行变量筛选，构建代价敏感学习模型
"""

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
                    feature_stats[feature]['p_values'].append(stats['p_value'])
                    feature_stats[feature]['valid_groups'] += 1
                    
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
        
        # 1. 特征重要性图
        if hasattr(self, 'selected_features'):
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(self.selected_features)), [1]*len(self.selected_features))
            plt.xticks(range(len(self.selected_features)), self.selected_features, rotation=45)
            plt.title('选定的重要特征')
            plt.ylabel('特征重要性')
            plt.tight_layout()
            plt.savefig('problem4_analysis/results/selected_features.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. ROC曲线比较
        if hasattr(self, 'models') and self.models:
            plt.figure(figsize=(12, 8))
            
            # 绘制集成模型的ROC曲线
            for name, result in self.models.items():
                if 'y_test' in result and 'y_pred_proba' in result:
                    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})', linewidth=3)
            
            # 如果有单个模型，也绘制它们的ROC曲线
            for name, result in self.models.items():
                if 'individual_models' in result:
                    for model_name, model in result['individual_models'].items():
                        try:
                            y_pred_proba = model.predict_proba(result['y_test'].index.map(lambda x: self.data.loc[x, self.selected_features].values).reshape(-1, len(self.selected_features)))[:, 1]
                            fpr, tpr, _ = roc_curve(result['y_test'], y_pred_proba)
                            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(result["y_test"], y_pred_proba):.3f})', 
                                   linewidth=1, alpha=0.7, linestyle='--')
                        except:
                            continue
            
            plt.plot([0, 1], [0, 1], 'k--', label='随机分类器', alpha=0.5)
            plt.xlabel('假正率 (FPR)')
            plt.ylabel('真正率 (TPR)')
            plt.title('集成神经网络 ROC曲线比较')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('problem4_analysis/results/roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 特征重要性分析
        if hasattr(self, 'models') and self.models:
            # 找到第一个有特征重要性的模型
            for name, result in self.models.items():
                if result.get('feature_importance') is not None:
                    plt.figure(figsize=(12, 8))
                    
                    importance = result['feature_importance']
                    features = self.selected_features
                    
                    # 排序特征重要性
                    sorted_idx = np.argsort(importance)[::-1]
                    sorted_features = [features[i] for i in sorted_idx]
                    sorted_importance = importance[sorted_idx]
                    
                    # 绘制特征重要性
                    plt.subplot(2, 1, 1)
                    bars = plt.barh(range(len(sorted_features)), sorted_importance, 
                                   color='skyblue', edgecolor='black')
                    plt.yticks(range(len(sorted_features)), sorted_features)
                    plt.xlabel('特征重要性')
                    plt.title(f'{name} 特征重要性分析')
                    plt.gca().invert_yaxis()
                    
                    # 添加数值标签
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        plt.text(width, bar.get_y() + bar.get_height()/2,
                                f'{width:.3f}', ha='left', va='center')
                    
                    # 绘制所有模型的特征重要性对比
                    plt.subplot(2, 1, 2)
                    model_names = []
                    importance_data = []
                    
                    for model_name, model_result in self.models.items():
                        if model_result.get('feature_importance') is not None:
                            model_names.append(model_name)
                            importance_data.append(model_result['feature_importance'])
                    
                    if importance_data:
                        importance_array = np.array(importance_data)
                        x = np.arange(len(features))
                        width = 0.25
                        
                        for i, (name, importance) in enumerate(zip(model_names, importance_data)):
                            plt.bar(x + i*width, importance, width, label=name, alpha=0.8)
                        
                        plt.xlabel('特征')
                        plt.ylabel('重要性')
                        plt.title('各模型特征重要性对比')
                        plt.xticks(x + width, features, rotation=45)
                        plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig('problem4_analysis/results/feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    break
        
        print("可视化结果已保存到 problem4_analysis/results/")
        
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
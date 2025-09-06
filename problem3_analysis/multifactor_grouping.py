"""
多因素分组模块 - 问题3
综合考虑身高、体重、年龄、BMI等多种因素进行分组
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MultifactorGrouping:
    """多因素分组器 - 考虑BMI、年龄、身高、体重等多种因素"""
    
    def __init__(self, max_groups: int = 5, min_samples_per_group: int = 20):
        """
        Parameters:
        - max_groups: 最大分组数量
        - min_samples_per_group: 每组最小样本数
        """
        self.max_groups = max_groups
        self.min_samples_per_group = min_samples_per_group
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.grouping_result = None
        
    def prepare_multifactor_features(self, data: pd.DataFrame, 
                                   time_predictor, risk_model) -> pd.DataFrame:
        """
        为多因素分组准备特征数据
        
        Parameters:
        - data: 原始数据
        - time_predictor: 时间预测模型
        - risk_model: 风险模型
        
        Returns:
        - feature_df: 包含多因素特征的数据框
        """
        features = []
        
        for _, patient in data.iterrows():
            # 基础特征
            bmi = patient['BMI']  # 使用BMI而不是孕妇BMI
            age = patient['年龄']
            height = patient['身高']
            weight = patient['体重']
            
            # 提取额外信息（如果存在）
            x_concentration = patient.get('X染色体浓度', 0.5)
            y_z_score = patient.get('Y染色体Z值', 0.0)
            x_z_score = patient.get('X染色体Z值', 0.0)
            chr18_z_score = patient.get('18号染色体Z值', 0.0)
            blood_draw_num = patient.get('检测抽血次数', 1)
            
            # 找到满足95%概率约束的最早时间
            min_time_for_95_percent = time_predictor.find_time_for_success_probability(
                bmi, age, height, weight, target_prob=risk_model.target_success_probability,
                x_concentration=x_concentration, y_z_score=y_z_score, x_z_score=x_z_score,
                chr18_z_score=chr18_z_score, blood_draw_num=blood_draw_num
            )
            
            if min_time_for_95_percent is not None:
                # 在约束范围内搜索最优时间
                from scipy.optimize import minimize_scalar
                
                def risk_objective(t):
                    risk_result = risk_model.calculate_multifactor_total_risk(
                        t, bmi, age, height, weight, time_predictor
                    )
                    return risk_result['total_risk']
                
                # 搜索范围：从满足约束的最早时间到合理的最大时间
                search_min = min_time_for_95_percent
                search_max = min(200, min_time_for_95_percent + 35)  # 最多延后5周
                
                try:
                    result = minimize_scalar(
                        risk_objective,
                        bounds=(search_min, search_max),
                        method='bounded'
                    )
                    
                    optimal_test_time = result.x
                    minimal_risk = result.fun
                    
                    # 计算该最优时间点的详细信息
                    risk_breakdown = risk_model.calculate_multifactor_total_risk(
                        optimal_test_time, bmi, age, height, weight, time_predictor
                    )
                    
                    # 计算多因素评分
                    multifactor_score = time_predictor.calculate_multifactor_score(
                        bmi, age, height, weight, x_concentration,
                        y_z_score, x_z_score, chr18_z_score, blood_draw_num
                    )
                    
                    features.append({
                        '孕妇代码': patient['孕妇代码'],
                        'BMI': bmi,
                        '年龄': age,
                        '身高': height,
                        '体重': weight,
                        'X染色体浓度': x_concentration,
                        'Y染色体Z值': y_z_score,
                        'X染色体Z值': x_z_score,
                        '18号染色体Z值': chr18_z_score,
                        '检测抽血次数': blood_draw_num,
                        # 基础特征
                        'BMI平方': bmi ** 2,
                        'BMI_age_interaction': bmi * age,
                        '身高体重比': height / weight if weight > 0 else 0,
                        'BMI_height_interaction': bmi * height,
                        'BMI_weight_interaction': bmi * weight,
                        'age_height_interaction': age * height,
                        'age_weight_interaction': age * weight,
                        # 时间相关特征
                        '满足95%约束的最早时间': min_time_for_95_percent,
                        '满足95%约束的最早周数': min_time_for_95_percent / 7,
                        '最优检测时间': optimal_test_time,
                        '最优检测周数': optimal_test_time / 7,
                        '最小总风险': minimal_risk,
                        '最优时点成功概率': risk_breakdown['success_probability'],
                        '最优时点检测失败风险': risk_breakdown['detection_failure_risk'],
                        '最优时点延误风险': risk_breakdown['delay_risk'],
                        '满足约束': risk_breakdown['satisfies_constraint'],
                        # 多因素评分
                        '多因素综合评分': multifactor_score['total_score'],
                        'BMI评分': multifactor_score['bmi_score'],
                        '年龄评分': multifactor_score['age_score'],
                        '身高评分': multifactor_score['height_score'],
                        '体重评分': multifactor_score['weight_score'],
                        '质量评分': multifactor_score['quality_score'],
                        'X染色体浓度评分': multifactor_score['x_conc_score'],
                        # 复合指标
                        '风险敏感指数': minimal_risk / (optimal_test_time / 7),
                        'BMI_风险_interaction': bmi * minimal_risk,
                        '多因素_风险_interaction': multifactor_score['total_score'] * minimal_risk
                    })
                    
                except Exception as e:
                    print(f"患者 {patient.get('孕妇代码', 'Unknown')} 优化失败: {e}")
                    # 使用备用方案
                    fallback_time = min_time_for_95_percent
                    fallback_risk = risk_objective(fallback_time)
                    risk_breakdown = risk_model.calculate_multifactor_total_risk(
                        fallback_time, bmi, age, height, weight, time_predictor
                    )
                    
                    multifactor_score = time_predictor.calculate_multifactor_score(
                        bmi, age, height, weight, x_concentration,
                        y_z_score, x_z_score, chr18_z_score, blood_draw_num
                    )
                    
                    features.append({
                        '孕妇代码': patient['孕妇代码'],
                        'BMI': bmi,
                        '年龄': age,
                        '身高': height,
                        '体重': weight,
                        'X染色体浓度': x_concentration,
                        'Y染色体Z值': y_z_score,
                        'X染色体Z值': x_z_score,
                        '18号染色体Z值': chr18_z_score,
                        '检测抽血次数': blood_draw_num,
                        'BMI平方': bmi ** 2,
                        'BMI_age_interaction': bmi * age,
                        '身高体重比': height / weight if weight > 0 else 0,
                        'BMI_height_interaction': bmi * height,
                        'BMI_weight_interaction': bmi * weight,
                        'age_height_interaction': age * height,
                        'age_weight_interaction': age * weight,
                        '满足95%约束的最早时间': min_time_for_95_percent,
                        '满足95%约束的最早周数': min_time_for_95_percent / 7,
                        '最优检测时间': fallback_time,
                        '最优检测周数': fallback_time / 7,
                        '最小总风险': fallback_risk,
                        '最优时点成功概率': risk_breakdown['success_probability'],
                        '最优时点检测失败风险': risk_breakdown['detection_failure_risk'],
                        '最优时点延误风险': risk_breakdown['delay_risk'],
                        '满足约束': risk_breakdown['satisfies_constraint'],
                        '多因素综合评分': multifactor_score['total_score'],
                        'BMI评分': multifactor_score['bmi_score'],
                        '年龄评分': multifactor_score['age_score'],
                        '身高评分': multifactor_score['height_score'],
                        '体重评分': multifactor_score['weight_score'],
                        '质量评分': multifactor_score['quality_score'],
                        'X染色体浓度评分': multifactor_score['x_conc_score'],
                        '风险敏感指数': fallback_risk / (fallback_time / 7),
                        'BMI_风险_interaction': bmi * fallback_risk,
                        '多因素_风险_interaction': multifactor_score['total_score'] * fallback_risk
                    })
            else:
                print(f"患者 {patient.get('孕妇代码', 'Unknown')} 无法在合理时间内达到95%概率要求")
        
        return pd.DataFrame(features)
    
    def optimize_multifactor_grouping(self, feature_df: pd.DataFrame) -> Dict:
        """
        使用多因素特征进行分组优化
        
        Parameters:
        - feature_df: 特征数据框
        
        Returns:
        - grouping_result: 分组优化结果
        """
        # 选择用于分组的特征
        grouping_features = [
            'BMI', '年龄', '身高', '体重', 'BMI平方', 'BMI_age_interaction',
            '身高体重比', 'BMI_height_interaction', 'BMI_weight_interaction',
            'age_height_interaction', 'age_weight_interaction',
            '多因素综合评分', 'BMI评分', '年龄评分', '身高评分', '体重评分',
            '质量评分', 'X染色体浓度评分', '风险敏感指数', '多因素_风险_interaction'
        ]
        
        # 确保所有特征列都存在
        available_features = [f for f in grouping_features if f in feature_df.columns]
        X = feature_df[available_features]
        
        print(f"可用特征数: {len(available_features)}")
        print(f"数据样本数: {len(feature_df)}")
        print(f"特征形状: {X.shape}")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 目标变量：最优检测周数
        y = feature_df['最优检测周数']
        
        # 使用决策树回归进行分组
        print("使用决策树回归进行多因素分组...")
        
        # 处理缺失值
        valid_mask = ~pd.isna(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        print(f"有效样本数: {len(X_valid)}")
        print(f"目标变量范围: {y_valid.min():.1f} - {y_valid.max():.1f}")
        
        if len(X_valid) < 10:
            raise ValueError("数据不足以进行分组")
        
        # 训练决策树回归
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import cross_val_score
        
        best_score = -np.inf
        best_model = None
        best_max_depth = None
        
        # 尝试不同的最大深度
        for max_depth in range(2, min(10, len(X_valid) // 10 + 1)):
            try:
                dt = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=max(2, len(X_valid) // 20),
                    min_samples_leaf=max(1, len(X_valid) // 50),
                    random_state=42
                )
                
                # 使用交叉验证评估（负均方误差，越大越好）
                scores = cross_val_score(dt, X_valid, y_valid, cv=min(5, len(X_valid) // 10), scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                
                print(f"  最大深度 {max_depth}: 负MSE={avg_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = dt
                    best_max_depth = max_depth
                    
            except Exception as e:
                print(f"  最大深度 {max_depth} 失败: {e}")
                continue
        
        if best_model is None:
            # 使用默认参数
            best_model = DecisionTreeRegressor(max_depth=3, random_state=42)
            best_max_depth = 3
            print("使用默认决策树参数")
        
        # 训练最终模型
        best_model.fit(X_valid, y_valid)
        
        # 预测所有样本
        y_pred = best_model.predict(X)
        
        # 基于预测值进行分组（使用分位数方法）
        n_groups = min(self.max_groups, 5)
        labels = pd.qcut(y_pred, q=n_groups, labels=False, duplicates='drop')
        
        # 处理分组失败的情况
        if len(np.unique(labels)) < 2:
            labels = pd.cut(y_pred, bins=n_groups, labels=False, duplicates='drop')
        
        print(f"决策树分组完成: 最大深度={best_max_depth}, 交叉验证负MSE={best_score:.3f}")
        print(f"分组分布: {np.bincount(labels)}")
        
        # 保存决策树模型
        self.clustering_model = best_model
        
        # 创建分组映射
        groups = {}
        group_stats = {}
        
        unique_labels = np.unique(labels)
        for i in unique_labels:
            mask = labels == i
            group_data = feature_df[mask]
            
            if len(group_data) >= self.min_samples_per_group:
                group_name = f"Group_{i+1}"
                groups[group_name] = {
                    'indices': feature_df[mask].index.tolist(),
                    'data': group_data,
                    'cluster_id': i
                }
                
                # 计算组统计信息
                group_stats[group_name] = {
                    'sample_size': len(group_data),
                    'bmi_range': (group_data['BMI'].min(), group_data['BMI'].max()),
                    'bmi_mean': group_data['BMI'].mean(),
                    'age_range': (group_data['年龄'].min(), group_data['年龄'].max()),
                    'age_mean': group_data['年龄'].mean(),
                    'height_range': (group_data['身高'].min(), group_data['身高'].max()),
                    'height_mean': group_data['身高'].mean(),
                    'weight_range': (group_data['体重'].min(), group_data['体重'].max()),
                    'weight_mean': group_data['体重'].mean(),
                    'optimal_test_time_mean': group_data['最优检测周数'].mean(),
                    'optimal_test_time_std': group_data['最优检测周数'].std(),
                    'minimal_risk_mean': group_data['最小总风险'].mean(),
                    'constraint_satisfaction_rate': group_data['满足约束'].mean(),
                    'success_probability_mean': group_data['最优时点成功概率'].mean(),
                    'multifactor_score_mean': group_data['多因素综合评分'].mean()
                }
        
        # 后处理：重新平衡样本分布
        groups, group_stats = self._rebalance_multifactor_groups(groups, group_stats, feature_df)
        
        self.grouping_result = {
            'clustering_model': best_model,
            'best_n_clusters': len(unique_labels),
            'best_method': 'DecisionTree',
            'best_score': best_score,
            'groups': groups,
            'group_stats': group_stats,
            'feature_names': available_features,
            'scaler': self.scaler
        }
        
        return self.grouping_result
    
    def _rebalance_multifactor_groups(self, groups: Dict, group_stats: Dict, 
                                    feature_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        重新平衡多因素分组样本分布
        
        Parameters:
        - groups: 原始分组
        - group_stats: 分组统计信息
        - feature_df: 特征数据框
        
        Returns:
        - (rebalanced_groups, rebalanced_stats): 重新平衡后的分组和统计信息
        """
        # 检查样本分布
        sample_sizes = [stats['sample_size'] for stats in group_stats.values()]
        min_samples = min(sample_sizes) if sample_sizes else 0
        max_samples = max(sample_sizes) if sample_sizes else 0
        
        # 如果样本分布过于不均匀，进行重新平衡
        if (max_samples > min_samples * 3 and min_samples < self.min_samples_per_group) or len(groups) < 3:
            print(f"检测到样本分布不均匀，使用多因素分位数重新分组...")
            print(f"原始分布: {sample_sizes}")
            
            # 使用多因素综合评分进行重新分组
            multifactor_scores = feature_df['多因素综合评分'].values
            multifactor_sorted = np.sort(multifactor_scores)
            
            # 计算分位数点
            n_samples = len(multifactor_scores)
            target_groups = min(5, max(3, n_samples // self.min_samples_per_group))
            
            quantile_points = np.linspace(0, 1, target_groups + 1)
            score_quantiles = np.quantile(multifactor_scores, quantile_points)
            
            # 重新分配样本到更平衡的组
            rebalanced_groups = {}
            rebalanced_stats = {}
            
            for i in range(target_groups):
                group_name = f"Group_{i+1}"
                
                if i == 0:
                    mask = multifactor_scores <= score_quantiles[1]
                elif i == target_groups - 1:
                    mask = multifactor_scores > score_quantiles[i]
                else:
                    mask = (multifactor_scores > score_quantiles[i]) & (multifactor_scores <= score_quantiles[i+1])
                
                group_data = feature_df[mask]
                
                if len(group_data) > 0:
                    rebalanced_groups[group_name] = {
                        'indices': feature_df[mask].index.tolist(),
                        'data': group_data,
                        'cluster_id': i
                    }
                    
                    rebalanced_stats[group_name] = {
                        'sample_size': len(group_data),
                        'bmi_range': (group_data['BMI'].min(), group_data['BMI'].max()),
                        'bmi_mean': group_data['BMI'].mean(),
                        'age_range': (group_data['年龄'].min(), group_data['年龄'].max()),
                        'age_mean': group_data['年龄'].mean(),
                        'height_range': (group_data['身高'].min(), group_data['身高'].max()),
                        'height_mean': group_data['身高'].mean(),
                        'weight_range': (group_data['体重'].min(), group_data['体重'].max()),
                        'weight_mean': group_data['体重'].mean(),
                        'optimal_test_time_mean': group_data['最优检测周数'].mean(),
                        'optimal_test_time_std': group_data['最优检测周数'].std(),
                        'minimal_risk_mean': group_data['最小总风险'].mean(),
                        'constraint_satisfaction_rate': group_data['满足约束'].mean(),
                        'success_probability_mean': group_data['最优时点成功概率'].mean(),
                        'multifactor_score_mean': group_data['多因素综合评分'].mean()
                    }
            
            new_sample_sizes = [stats['sample_size'] for stats in rebalanced_stats.values()]
            print(f"重新平衡后分布: {new_sample_sizes}")
            print(f"多因素评分分位数点: {score_quantiles}")
            
            return rebalanced_groups, rebalanced_stats
        else:
            print(f"样本分布相对均匀，保持原分组")
            return groups, group_stats
    
    def refine_multifactor_grouping_with_bmi_bounds(self, feature_df: pd.DataFrame) -> Dict:
        """
        基于多因素分组结果，细化BMI分组边界
        
        Parameters:
        - feature_df: 特征数据框
        
        Returns:
        - refined_groups: 细化的分组结果
        """
        if self.grouping_result is None:
            raise ValueError("请先运行optimize_multifactor_grouping")
        
        # 首先收集所有组的BMI信息
        group_bmi_info = []
        for group_name, group_info in self.grouping_result['groups'].items():
            group_data = group_info['data']
            bmi_min = group_data['BMI'].min()
            bmi_max = group_data['BMI'].max()
            bmi_mean = group_data['BMI'].mean()
            
            group_bmi_info.append({
                'group_name': group_name,
                'group_data': group_data,
                'bmi_min': bmi_min,
                'bmi_max': bmi_max,
                'bmi_mean': bmi_mean,
                'sample_size': len(group_data)
            })
        
        # 按BMI均值排序
        group_bmi_info.sort(key=lambda x: x['bmi_mean'])
        
        # 使用严格的分段约束方法
        refined_groups = {}
        
        # 获取全局BMI范围
        all_bmi_values = feature_df['BMI'].values
        global_bmi_min = all_bmi_values.min()
        global_bmi_max = all_bmi_values.max()
        
        # 计算严格的分段边界点
        n_groups = len(group_bmi_info)
        boundary_points = np.linspace(global_bmi_min, global_bmi_max, n_groups + 1)
        
        # 为每个组重新定义边界，确保严格相邻且不交
        for i, group_info in enumerate(group_bmi_info):
            group_name = group_info['group_name']
            group_data = group_info['group_data']
            
            # 使用计算出的边界点
            new_bmi_min = boundary_points[i]
            new_bmi_max = boundary_points[i + 1]
            
            # 对于最后一组，确保包含最大值
            if i == n_groups - 1:
                new_bmi_max = global_bmi_max
            
            # 计算该组的推荐检测时间范围
            optimal_times = group_data['最优检测周数']
            recommended_time = optimal_times.mean()
            time_std = optimal_times.std()
            
            refined_groups[group_name] = {
                'bmi_range': (new_bmi_min, new_bmi_max),
                'bmi_interval': f"[{new_bmi_min:.1f}, {new_bmi_max:.1f}]",
                'bmi_mean': group_info['bmi_mean'],
                'age_mean': group_data['年龄'].mean(),
                'height_mean': group_data['身高'].mean(),
                'weight_mean': group_data['体重'].mean(),
                'multifactor_score_mean': group_data['多因素综合评分'].mean(),
                'sample_size': group_info['sample_size'],
                'recommended_test_time_weeks': recommended_time,
                'test_time_std': time_std,
                'test_time_range': (recommended_time - time_std, recommended_time + time_std),
                'expected_minimal_risk': group_data['最小总风险'].mean(),
                'risk_std': group_data['最小总风险'].std(),
                'constraint_satisfaction_rate': group_data['满足约束'].mean(),
                'average_success_probability': group_data['最优时点成功概率'].mean(),
                'group_data': group_data
            }
        
        # 验证分段约束
        print(f"BMI分段边界点: {boundary_points}")
        
        return refined_groups
    
    def generate_multifactor_grouping_rules(self, refined_groups: Dict) -> List[Dict]:
        """
        生成多因素分组规则
        
        Parameters:
        - refined_groups: 细化的分组结果
        
        Returns:
        - grouping_rules: 分组规则列表
        """
        rules = []
        
        for i, (group_name, group_info) in enumerate(refined_groups.items()):
            rule = {
                'group_id': i + 1,
                'group_name': f"多因素组{i+1}",
                'bmi_lower_bound': group_info['bmi_range'][0],
                'bmi_upper_bound': group_info['bmi_range'][1],
                'bmi_interval_description': group_info['bmi_interval'],
                'bmi_mean': group_info['bmi_mean'],
                'age_mean': group_info['age_mean'],
                'height_mean': group_info['height_mean'],
                'weight_mean': group_info['weight_mean'],
                'multifactor_score_mean': group_info['multifactor_score_mean'],
                'sample_size': group_info['sample_size'],
                'recommended_test_time_weeks': round(group_info['recommended_test_time_weeks'], 1),
                'recommended_test_time_days': round(group_info['recommended_test_time_weeks'] * 7),
                'expected_risk': round(group_info['expected_minimal_risk'], 3),
                'clinical_recommendation': self._get_multifactor_clinical_recommendation(
                    group_info['recommended_test_time_weeks'],
                    group_info['multifactor_score_mean']
                )
            }
            rules.append(rule)
        
        return rules
    
    def _get_multifactor_clinical_recommendation(self, test_time_weeks: float, 
                                               multifactor_score: float) -> str:
        """生成多因素临床建议文本"""
        if test_time_weeks <= 12:
            timing = "早期"
        elif test_time_weeks <= 18:
            timing = "中期"
        else:
            timing = "中晚期"
        
        if multifactor_score <= 0.3:
            complexity = "低风险"
        elif multifactor_score <= 0.6:
            complexity = "中等风险"
        else:
            complexity = "高风险"
        
        return f"{timing}检测，{complexity}，建议个体化监测"


if __name__ == "__main__":
    # 测试多因素分组
    print("多因素分组模块测试")
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        '孕妇代码': [f'T{i:03d}' for i in range(100)],
        '孕妇BMI': np.random.normal(32, 5, 100),
        '年龄': np.random.normal(30, 4, 100),
        '身高': np.random.normal(165, 8, 100),
        '体重': np.random.normal(70, 10, 100),
        'X染色体浓度': np.random.normal(0.5, 0.1, 100),
        'Y染色体Z值': np.random.normal(0, 1, 100),
        'X染色体Z值': np.random.normal(0, 1, 100),
        '18号染色体Z值': np.random.normal(0, 1, 100),
        '检测抽血次数': np.random.randint(1, 4, 100)
    })
    
    grouping_optimizer = MultifactorGrouping()
    print(f"初始化完成，最大分组数: {grouping_optimizer.max_groups}")
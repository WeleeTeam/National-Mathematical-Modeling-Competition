"""
决策树分组模块
使用决策树算法优化BMI分组策略
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DecisionTreeGrouping:
    """基于决策树的BMI分组优化器"""
    
    def __init__(self, max_groups: int = 5, min_samples_per_group: int = 20):
        """
        Parameters:
        - max_groups: 最大分组数量（增加到5组以满足BMI分段约束）
        - min_samples_per_group: 每组最小样本数（减少到20以允许更多分组）
        """
        self.max_groups = max_groups
        self.min_samples_per_group = min_samples_per_group
        self.tree_model = None
        self.grouping_result = None
        
    def prepare_features_for_grouping(self, data: pd.DataFrame, 
                                    time_predictor, risk_model) -> pd.DataFrame:
        """
        为决策树准备特征数据（使用新的风险优化逻辑）
        
        Parameters:
        - data: 原始数据
        - time_predictor: 时间预测模型
        - risk_model: 风险模型
        
        Returns:
        - feature_df: 包含特征和目标变量的数据框
        """
        features = []
        
        for _, patient in data.iterrows():
            # 基础特征
            bmi = patient['孕妇BMI']
            age = patient['年龄']
            height = patient['身高']
            weight = patient['体重']
            
            # 找到满足95%概率约束的最早时间
            min_time_for_95_percent = time_predictor.find_time_for_success_probability(
                bmi, age, height, weight, target_prob=risk_model.target_success_probability
            )
            
            if min_time_for_95_percent is not None:
                # 在约束范围内搜索最优时间
                from scipy.optimize import minimize_scalar
                
                def risk_objective(t):
                    risk_result = risk_model.calculate_total_risk(
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
                    risk_breakdown = risk_model.calculate_total_risk(
                        optimal_test_time, bmi, age, height, weight, time_predictor
                    )
                    
                    features.append({
                        '孕妇代码': patient['孕妇代码'],
                        'BMI': bmi,
                        '年龄': age,
                        '身高': height,
                        '体重': weight,
                        'BMI平方': bmi ** 2,
                        'BMI_age_interaction': bmi * age,
                        '身高体重比': height / weight if weight > 0 else 0,
                        '满足95%约束的最早时间': min_time_for_95_percent,
                        '满足95%约束的最早周数': min_time_for_95_percent / 7,
                        '最优检测时间': optimal_test_time,
                        '最优检测周数': optimal_test_time / 7,
                        '最小总风险': minimal_risk,
                        '最优时点成功概率': risk_breakdown['success_probability'],
                        '最优时点检测失败风险': risk_breakdown['detection_failure_risk'],
                        '最优时点延误风险': risk_breakdown['delay_risk'],
                        '满足约束': risk_breakdown['satisfies_constraint'],
                        '风险敏感指数': minimal_risk / (optimal_test_time / 7),  # 复合指标：单位时间风险
                        'BMI_风险_interaction': bmi * minimal_risk  # BMI与风险的交互项
                    })
                    
                except Exception as e:
                    print(f"患者 {patient.get('孕妇代码', 'Unknown')} 优化失败: {e}")
                    # 使用备用方案
                    fallback_time = min_time_for_95_percent
                    fallback_risk = risk_objective(fallback_time)
                    risk_breakdown = risk_model.calculate_total_risk(
                        fallback_time, bmi, age, height, weight, time_predictor
                    )
                    
                    features.append({
                        '孕妇代码': patient['孕妇代码'],
                        'BMI': bmi,
                        '年龄': age,
                        '身高': height,
                        '体重': weight,
                        'BMI平方': bmi ** 2,
                        'BMI_age_interaction': bmi * age,
                        '身高体重比': height / weight if weight > 0 else 0,
                        '满足95%约束的最早时间': min_time_for_95_percent,
                        '满足95%约束的最早周数': min_time_for_95_percent / 7,
                        '最优检测时间': fallback_time,
                        '最优检测周数': fallback_time / 7,
                        '最小总风险': fallback_risk,
                        '最优时点成功概率': risk_breakdown['success_probability'],
                        '最优时点检测失败风险': risk_breakdown['detection_failure_risk'],
                        '最优时点延误风险': risk_breakdown['delay_risk'],
                        '满足约束': risk_breakdown['satisfies_constraint'],
                        '风险敏感指数': fallback_risk / (fallback_time / 7),
                        'BMI_风险_interaction': bmi * fallback_risk
                    })
            else:
                print(f"患者 {patient.get('孕妇代码', 'Unknown')} 无法在合理时间内达到95%概率要求")
        
        return pd.DataFrame(features)
    
    def optimize_grouping_with_decision_tree(self, feature_df: pd.DataFrame) -> Dict:
        """
        使用决策树优化分组策略
        
        Parameters:
        - feature_df: 特征数据框
        
        Returns:
        - grouping_result: 分组优化结果
        """
        # 选择用于分组的特征（使用新的特征列）
        grouping_features = ['BMI', '年龄', '身高', '体重', 'BMI平方', 'BMI_age_interaction', 
                           '身高体重比', '风险敏感指数', 'BMI_风险_interaction']
        
        # 确保所有特征列都存在
        available_features = [f for f in grouping_features if f in feature_df.columns]
        X = feature_df[available_features]
        
        # 目标变量：最优检测周数（这是我们想要预测和分组的目标）
        y = feature_df['最优检测周数']

        # 使用分层采样平衡样本分布
        from sklearn.model_selection import StratifiedKFold
        
        # 创建BMI分层标签（用于分层采样）
        bmi_quantiles = np.quantile(feature_df['BMI'], [0.33, 0.67])
        bmi_strata = np.where(feature_df['BMI'] <= bmi_quantiles[0], 0,
                             np.where(feature_df['BMI'] <= bmi_quantiles[1], 1, 2))
        
        # 尝试不同的树深度，找到最优分组
        best_score = -np.inf
        best_tree = None
        best_depth = None
        
        # 限制树深度为2-3，避免过拟合
        for max_depth in range(2, min(self.max_groups + 1, 4)):
            # 创建决策树回归器，使用更保守的参数
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=max(self.min_samples_per_group, len(feature_df) // 20),  # 动态调整最小样本数
                min_samples_split=max(self.min_samples_per_group * 2, len(feature_df) // 10),
                max_features='sqrt',  # 限制特征数量，防止过拟合
                random_state=42,
                ccp_alpha=0.01  # 添加剪枝参数
            )
            
            # 使用分层交叉验证评估
            try:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(tree, X, y, cv=skf.split(X, bmi_strata), 
                                       scoring='neg_mean_squared_error')
                avg_score = scores.mean()
            except:
                # 如果分层采样失败，使用普通交叉验证
                scores = cross_val_score(tree, X, y, cv=3, scoring='neg_mean_squared_error')
                avg_score = scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                best_tree = tree
                best_depth = max_depth
        
        # 用最佳树拟合完整数据
        best_tree.fit(X, y)
        self.tree_model = best_tree
        
        # 获取叶节点分组
        leaf_ids = best_tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        # 创建分组映射
        groups = {}
        group_stats = {}
        
        for i, leaf_id in enumerate(unique_leaves):
            mask = leaf_ids == leaf_id
            group_data = feature_df[mask]
            
            group_name = f"Group_{i+1}"
            groups[group_name] = {
                'indices': feature_df[mask].index.tolist(),
                'data': group_data,
                'leaf_id': leaf_id
            }
            
            # 计算组统计信息
            group_stats[group_name] = {
                'sample_size': len(group_data),
                'bmi_range': (group_data['BMI'].min(), group_data['BMI'].max()),
                'bmi_mean': group_data['BMI'].mean(),
                'age_range': (group_data['年龄'].min(), group_data['年龄'].max()),
                'optimal_test_time_mean': group_data['最优检测周数'].mean(),
                'optimal_test_time_std': group_data['最优检测周数'].std(),
                'minimal_risk_mean': group_data['最小总风险'].mean(),
                'constraint_satisfaction_rate': group_data['满足约束'].mean(),
                'success_probability_mean': group_data['最优时点成功概率'].mean()
            }
        
        # 后处理：重新平衡样本分布
        groups, group_stats = self._rebalance_groups(groups, group_stats, feature_df)

        self.grouping_result = {
            'tree_model': best_tree,
            'best_depth': best_depth,
            'best_cv_score': best_score,
            'groups': groups,
            'group_stats': group_stats,
            'feature_names': available_features,  # 保存实际使用的特征名称
            'feature_importance': dict(zip(available_features, best_tree.feature_importances_)),
            'tree_rules': export_text(best_tree, feature_names=available_features)
        }
        
        return self.grouping_result
    
    def _rebalance_groups(self, groups: Dict, group_stats: Dict, feature_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        重新平衡样本分布，使用基于分位数的分组方法确保满足BMI分段约束
        
        Parameters:
        - groups: 原始分组
        - group_stats: 分组统计信息
        - feature_df: 特征数据框
        
        Returns:
        - (rebalanced_groups, rebalanced_stats): 重新平衡后的分组和统计信息
        """
        # 检查样本分布
        sample_sizes = [stats['sample_size'] for stats in group_stats.values()]
        min_samples = min(sample_sizes)
        max_samples = max(sample_sizes)
        
        # 如果样本分布过于不均匀或组数不足，进行重新平衡
        if (max_samples > min_samples * 3 and min_samples < self.min_samples_per_group) or len(groups) < 4:
            print(f"检测到样本分布不均匀或组数不足，使用分位数分组...")
            print(f"原始分布: {sample_sizes}")
            
            # 使用基于BMI的分位数重新分组，确保满足分段约束
            bmi_values = feature_df['BMI'].values
            bmi_sorted = np.sort(bmi_values)
            
            # 计算分位数点，确保每组都有足够的样本
            n_samples = len(bmi_values)
            target_groups = min(5, max(3, n_samples // self.min_samples_per_group))
            
            # 计算分位数
            quantile_points = np.linspace(0, 1, target_groups + 1)
            bmi_quantiles = np.quantile(bmi_values, quantile_points)
            
            # 重新分配样本到更平衡的组
            rebalanced_groups = {}
            rebalanced_stats = {}
            
            for i in range(target_groups):
                group_name = f"Group_{i+1}"
                
                if i == 0:
                    mask = bmi_values <= bmi_quantiles[1]
                elif i == target_groups - 1:
                    mask = bmi_values > bmi_quantiles[i]
                else:
                    mask = (bmi_values > bmi_quantiles[i]) & (bmi_values <= bmi_quantiles[i+1])
                
                group_data = feature_df[mask]
                
                if len(group_data) > 0:
                    rebalanced_groups[group_name] = {
                        'indices': feature_df[mask].index.tolist(),
                        'data': group_data,
                        'leaf_id': i
                    }
                    
                    rebalanced_stats[group_name] = {
                        'sample_size': len(group_data),
                        'bmi_range': (group_data['BMI'].min(), group_data['BMI'].max()),
                        'bmi_mean': group_data['BMI'].mean(),
                        'age_range': (group_data['年龄'].min(), group_data['年龄'].max()),
                        'optimal_test_time_mean': group_data['最优检测周数'].mean(),
                        'optimal_test_time_std': group_data['最优检测周数'].std(),
                        'minimal_risk_mean': group_data['最小总风险'].mean(),
                        'constraint_satisfaction_rate': group_data['满足约束'].mean(),
                        'success_probability_mean': group_data['最优时点成功概率'].mean()
                    }
            
            new_sample_sizes = [stats['sample_size'] for stats in rebalanced_stats.values()]
            print(f"重新平衡后分布: {new_sample_sizes}")
            print(f"BMI分位数点: {bmi_quantiles}")
            
            return rebalanced_groups, rebalanced_stats
        else:
            print(f"样本分布相对均匀，保持原分组")
            return groups, group_stats

    def refine_grouping_with_bmi_bounds(self, feature_df: pd.DataFrame) -> Dict:
        """
        基于决策树结果，细化BMI分组边界（确保满足三个约束条件）
        使用严格的分段约束方法，确保相邻且不交
        
        Parameters:
        - feature_df: 特征数据框
        
        Returns:
        - refined_groups: 细化的分组结果
        """
        if self.grouping_result is None:
            raise ValueError("请先运行optimize_grouping_with_decision_tree")
        
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
        bmi_groups = []  # 用于约束检查
        
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
            
            # 收集BMI分组信息用于约束检查
            bmi_groups.append((new_bmi_min, new_bmi_max))
        
        # 验证分段约束
        print(f"BMI分段边界点: {boundary_points}")
        print(f"分组边界: {bmi_groups}")
        
        return refined_groups
    
    def validate_bmi_segmentation_constraints(self, refined_groups: Dict) -> Dict:
        """
        验证BMI分段约束是否满足
        
        Parameters:
        - refined_groups: 细化的分组结果
        
        Returns:
        - validation_result: 约束验证结果
        """
        # 提取BMI分组边界
        bmi_groups = []
        for group_name, group_info in refined_groups.items():
            bmi_range = group_info['bmi_range']
            bmi_groups.append((bmi_range[0], bmi_range[1]))
        
        # 按边界排序
        bmi_groups.sort(key=lambda x: x[0])
        
        # 检查覆盖约束
        coverage_valid = True
        if bmi_groups:
            global_min = min(bmi_groups[0][0], 20.0)  # 假设BMI最小值为20
            global_max = max(bmi_groups[-1][1], 50.0)  # 假设BMI最大值为50
            
            if bmi_groups[0][0] > global_min or bmi_groups[-1][1] < global_max:
                coverage_valid = False
        
        # 检查相邻约束
        adjacent_valid = True
        for i in range(len(bmi_groups) - 1):
            current_upper = bmi_groups[i][1]
            next_lower = bmi_groups[i + 1][0]
            
            # 相邻约束：当前组上界 = 下一组下界
            if abs(current_upper - next_lower) > 1e-6:
                adjacent_valid = False
                break
        
        # 检查单调性
        monotonic_valid = True
        boundaries = [bmi_groups[0][0]] + [group[1] for group in bmi_groups]
        for i in range(len(boundaries) - 1):
            if boundaries[i] >= boundaries[i + 1]:
                monotonic_valid = False
                break
        
        # 检查不交约束（相邻组不重叠）
        non_overlapping_valid = True
        for i in range(len(bmi_groups) - 1):
            current_upper = bmi_groups[i][1]
            next_lower = bmi_groups[i + 1][0]
            
            if current_upper > next_lower:
                non_overlapping_valid = False
                break
        
        overall_valid = coverage_valid and adjacent_valid and monotonic_valid and non_overlapping_valid
        
        validation_result = {
            'overall_valid': overall_valid,
            'coverage_valid': coverage_valid,
            'adjacent_valid': adjacent_valid,
            'monotonic_valid': monotonic_valid,
            'non_overlapping_valid': non_overlapping_valid,
            'bmi_groups': bmi_groups,
            'boundaries': boundaries
        }
        
        print(f"BMI分段约束验证结果:")
        print(f"  - 整体满足: {'✅' if overall_valid else '❌'}")
        print(f"  - 覆盖约束: {'✅' if coverage_valid else '❌'}")
        print(f"  - 相邻约束: {'✅' if adjacent_valid else '❌'}")
        print(f"  - 单调性约束: {'✅' if monotonic_valid else '❌'}")
        print(f"  - 不交约束: {'✅' if non_overlapping_valid else '❌'}")
        
        return validation_result
    
    def generate_grouping_rules(self, refined_groups: Dict) -> List[Dict]:
        """
        生成最终的分组规则
        
        Parameters:
        - refined_groups: 细化的分组结果
        
        Returns:
        - grouping_rules: 分组规则列表
        """
        rules = []
        
        for i, (group_name, group_info) in enumerate(refined_groups.items()):
            rule = {
                'group_id': i + 1,
                'group_name': f"BMI组{i+1}",
                'bmi_lower_bound': group_info['bmi_range'][0],
                'bmi_upper_bound': group_info['bmi_range'][1],
                'bmi_interval_description': group_info['bmi_interval'],
                'sample_size': group_info['sample_size'],
                'recommended_test_time_weeks': round(group_info['recommended_test_time_weeks'], 1),
                'recommended_test_time_days': round(group_info['recommended_test_time_weeks'] * 7),
                'expected_risk': round(group_info['expected_minimal_risk'], 3),
                'clinical_recommendation': self._get_clinical_recommendation(
                    group_info['recommended_test_time_weeks']
                )
            }
            rules.append(rule)
        
        return rules
    
    def _get_clinical_recommendation(self, test_time_weeks: float) -> str:
        """生成临床建议文本"""
        if test_time_weeks <= 12:
            return f"建议在{test_time_weeks:.1f}周检测，属于早期检测，风险较低"
        elif test_time_weeks <= 18:
            return f"建议在{test_time_weeks:.1f}周检测，属于中期检测，需要密切关注"
        else:
            return f"建议在{test_time_weeks:.1f}周检测，属于中晚期检测，建议提前准备"
    
    def visualize_decision_tree(self, save_path: str = None) -> plt.Figure:
        """可视化决策树"""
        if self.tree_model is None:
            raise ValueError("请先运行optimize_grouping_with_decision_tree")
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # 动态构建特征名称列表
        if hasattr(self, 'grouping_result') and 'feature_names' in self.grouping_result:
            feature_names = self.grouping_result['feature_names']
        else:
            feature_names = ['BMI', '年龄', '身高', '体重', 'BMI²', 'BMI×年龄', '身高体重比', '风险敏感指数', 'BMI_风险_interaction']
        
        plot_tree(self.tree_model, 
                 feature_names=feature_names,
                 filled=True, 
                 rounded=True,
                 fontsize=10,
                 ax=ax)
        
        plt.title('BMI分组决策树', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # 测试决策树分组
    print("决策树分组模块测试")
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        '孕妇代码': [f'T{i:03d}' for i in range(100)],
        '孕妇BMI': np.random.normal(32, 5, 100),
        '年龄': np.random.normal(30, 4, 100),
        '身高': np.random.normal(165, 8, 100),
        '体重': np.random.normal(70, 10, 100)
    })
    
    grouping_optimizer = DecisionTreeGrouping()
    print(f"初始化完成，最大分组数: {grouping_optimizer.max_groups}")
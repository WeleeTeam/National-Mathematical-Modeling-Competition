"""
简化决策树分组模块
使用决策树算法优化BMI分组策略
按新思路：以最早达标孕周天数作为分组依据，使用简化特征
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
    """基于简化决策树的BMI分组优化器"""
    
    def __init__(self, max_groups: int = 5, min_samples_per_group: int = 20):
        """
        Parameters:
        - max_groups: 最大分组数量
        - min_samples_per_group: 每组最小样本数
        """
        self.max_groups = max_groups
        self.min_samples_per_group = min_samples_per_group
        self.tree_model = None
        self.grouping_result = None
        
    def prepare_simplified_features_for_grouping(self, data: pd.DataFrame, 
                                           time_predictor, risk_model) -> pd.DataFrame:
        """
        为决策树准备简化特征数据
        新思路：以最早达标孕周天数作为分组依据，使用简化特征
        
        Parameters:
        - data: 原始数据
        - time_predictor: 简化时间预测模型
        - risk_model: 风险模型
        
        Returns:
        - feature_df: 包含简化特征和目标变量的数据框
        """
        features = []
        
        print("=== 准备简化特征数据 ===")
        
        for _, patient in data.iterrows():
            # 核心特征：只保留BMI作为主要决策变量
            bmi = patient['孕妇BMI']
            
            try:
                # 使用简化模型计算最早达标时间（只需BMI参数）
                earliest_达标时间 = time_predictor.solve_达标时间(bmi=bmi)
                
                if earliest_达标时间 is not None:
                    # 计算95%概率约束时间
                    constraint_time = time_predictor.find_time_for_success_probability(
                        bmi=bmi, target_prob=risk_model.target_success_probability
                    )
                    
                    # 在约束范围内搜索最优风险时间
                    if constraint_time is not None:
                        from scipy.optimize import minimize_scalar
                        
                        def risk_objective(t):
                            # 使用简化的风险计算方法，删除年龄、身高、体重参数
                            risk_result = risk_model.calculate_total_risk(
                                t, bmi, time_predictor
                            )
                            return risk_result['total_risk']
                        
                        # 搜索范围：从约束时间到合理的最大时间
                        search_min = constraint_time
                        search_max = min(220, constraint_time + 35)  # 最多延后5周
                        
                        try:
                            result = minimize_scalar(
                                risk_objective,
                                bounds=(search_min, search_max),
                                method='bounded'
                            )
                            
                            optimal_test_time = result.x
                            minimal_risk = result.fun
                            
                        except:
                            # 使用约束时间作为备用方案
                            optimal_test_time = constraint_time
                            minimal_risk = risk_objective(constraint_time)
                    else:
                        # 如果无法满足95%约束，使用达标时间
                        optimal_test_time = earliest_达标时间
                        minimal_risk = None
                    
                    # 保存简化特征（按您要求的参数）
                    features.append({
                        '孕妇代码': patient['孕妇代码'],
                        'BMI': bmi,  # 主要决策变量
                        '最早达标时间': earliest_达标时间,
                        '最早达标周数': earliest_达标时间 / 7,  # 主要分组依据
                        '95%约束时间': constraint_time,
                        '95%约束周数': constraint_time / 7 if constraint_time else None,
                        '最优检测时间': optimal_test_time,
                        '最优检测周数': optimal_test_time / 7,
                        '最小风险': minimal_risk if minimal_risk else 0,
                        '满足95%约束': constraint_time is not None
                    })
                    
                else:
                    print(f"患者 {patient.get('孕妇代码', 'Unknown')} 无法在合理时间内达标")
                    
            except Exception as e:
                print(f"患者 {patient.get('孕妇代码', 'Unknown')} 处理失败: {e}")
        
        print(f"成功处理 {len(features)} 个患者")
        return pd.DataFrame(features)
    
    def optimize_grouping_with_simplified_decision_tree(self, feature_df: pd.DataFrame) -> Dict:
        """
        使用简化决策树优化分组策略
        新思路：以最早达标周数作为分组目标，BMI作为主要特征
        
        Parameters:
        - feature_df: 特征数据框
        
        Returns:
        - grouping_result: 分组优化结果
        """
        print("=== 使用简化决策树进行分组优化 ===")
        
        # 简化特征选择：只使用BMI作为主要分组特征
        grouping_features = ['BMI']  # 按新思路只使用BMI作为决策变量
        
        X = feature_df[grouping_features]
        
        # 目标变量：最早达标周数（按您的要求作为分组依据）
        y = feature_df['最早达标周数']
        
        print(f"使用特征: {grouping_features}")
        print(f"目标变量: 最早达标周数")
        print(f"数据shape: {X.shape}")

        # 尝试不同的树深度，找到最优分组
        best_score = -np.inf
        best_tree = None
        best_depth = None
        
        print("正在搜索最优决策树深度...")
        
        # 限制树深度，适应简化特征
        for max_depth in range(2, min(self.max_groups + 1, 4)):
            # 创建决策树回归器，针对单特征优化参数
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=max(self.min_samples_per_group, len(feature_df) // 15),
                min_samples_split=max(self.min_samples_per_group * 2, len(feature_df) // 8),
                random_state=42,
                ccp_alpha=0.005  # 适度剪枝
            )
            
            # 使用交叉验证评估（简化版本）
            try:
                scores = cross_val_score(tree, X, y, cv=3, scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                
                print(f"深度 {max_depth}: CV分数 = {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_tree = tree
                    best_depth = max_depth
                    
            except Exception as e:
                print(f"深度 {max_depth} 评估失败: {e}")
        
        # 用最佳树拟合完整数据
        if best_tree is not None:
            best_tree.fit(X, y)
            self.tree_model = best_tree
            
            print(f"最佳决策树深度: {best_depth}, 分数: {best_score:.4f}")
            
            # 获取叶节点分组
            leaf_ids = best_tree.apply(X)
            unique_leaves = np.unique(leaf_ids)
            
            print(f"决策树生成 {len(unique_leaves)} 个叶节点")
            
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
                
                # 计算简化的组统计信息
                group_stats[group_name] = {
                    'sample_size': len(group_data),
                    'bmi_range': (group_data['BMI'].min(), group_data['BMI'].max()),
                    'bmi_mean': group_data['BMI'].mean(),
                    'earliest_达标时间_mean': group_data['最早达标周数'].mean(),  # 主要分组依据
                    'earliest_达标时间_std': group_data['最早达标周数'].std(),
                    'optimal_test_time_mean': group_data['最优检测周数'].mean(),
                    'optimal_test_time_std': group_data['最优检测周数'].std(),
                    'minimal_risk_mean': group_data['最小风险'].mean(),
                    'constraint_satisfaction_rate': group_data['满足95%约束'].mean()
                }
        
            # 后处理：重新平衡样本分布（如果需要）
            if len(groups) > 1:
                groups, group_stats = self._simplified_rebalance_groups(groups, group_stats, feature_df)

            self.grouping_result = {
                'tree_model': best_tree,
                'best_depth': best_depth,
                'best_cv_score': best_score,
                'groups': groups,
                'group_stats': group_stats,
                'feature_names': grouping_features,  # 使用简化的特征名称
                'feature_importance': dict(zip(grouping_features, best_tree.feature_importances_)),
                'total_groups': len(groups)
            }
            
            print(f"分组完成，共生成 {len(groups)} 个组")
            
        else:
            print("决策树拟合失败，使用备用分组方案")
            # 备用方案：基于达标时间的简单分组
            self.grouping_result = self._fallback_grouping_by_weeks(feature_df)
        
        return self.grouping_result
    
    def _simplified_rebalance_groups(self, groups: Dict, group_stats: Dict, 
                                   feature_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        简化的样本重平衡方法
        基于BMI范围确保分组合理
        """
        print("=== 简化样本重平衡 ===")
        
        # 检查样本分布是否需要调整
        group_sizes = [stats['sample_size'] for stats in group_stats.values()]
        min_size = min(group_sizes)
        max_size = max(group_sizes)
        
        if max_size / min_size > 2.0:  # 如果组间差异过大
            print("检测到样本分布不均匀，进行调整...")
            
            # 基于BMI均匀分组（备用方案）
            return self._fallback_grouping_by_weeks(feature_df, as_groups=True)
        else:
            print("样本分布较为均匀，保持原分组")
            return groups, group_stats

    def _fallback_grouping_by_weeks(self, feature_df: pd.DataFrame, as_groups: bool = False) -> Dict:
        """
        基于达标时间的备用分组方案
        直接按达标周数区间进行分组
        """
        print("=== 使用备用分组方案：按达标时间分组 ===")
        
        # 按达标周数分组
        weeks = feature_df['最早达标周数']
        
        # 定义分组区间（根据达标时间分布）
        quantiles = np.quantile(weeks.dropna(), [0.25, 0.5, 0.75])
        
        def assign_group(week):
            if pd.isna(week):
                return 'Group_Unknown'
            elif week <= quantiles[0]:
                return 'Group_1'  # 早期达标组
            elif week <= quantiles[1]:
                return 'Group_2'  # 中早期达标组
            elif week <= quantiles[2]:
                return 'Group_3'  # 中期达标组
            else:
                return 'Group_4'  # 晚期达标组
        
        feature_df['Group'] = feature_df['最早达标周数'].apply(assign_group)
        
        # 构建分组结果
        groups = {}
        group_stats = {}
        
        for group_name in feature_df['Group'].unique():
            if group_name == 'Group_Unknown':
                continue
                
            group_data = feature_df[feature_df['Group'] == group_name]
            
            groups[group_name] = {
                'indices': group_data.index.tolist(),
                'data': group_data,
                'assignment_method': 'weeks_based'
            }
            
            group_stats[group_name] = {
                'sample_size': len(group_data),
                'bmi_range': (group_data['BMI'].min(), group_data['BMI'].max()),
                'bmi_mean': group_data['BMI'].mean(),
                'earliest_达标时间_mean': group_data['最早达标周数'].mean(),
                'earliest_达标时间_std': group_data['最早达标周数'].std(),
                'optimal_test_time_mean': group_data['最优检测周数'].mean(),
                'constraint_satisfaction_rate': group_data['满足95%约束'].mean()
            }
        
        if as_groups:
            return groups, group_stats
        else:
            return {
                'tree_model': None,
                'best_depth': None,
                'best_cv_score': None,
                'groups': groups,
                'group_stats': group_stats,
                'feature_names': ['BMI'],
                'feature_importance': {'BMI': 1.0},
                'total_groups': len(groups),
                'method': 'fallback_weeks_based'
            }

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


if __name__ == "__main__":
    # 测试模块
    test_data = pd.DataFrame({
        '孕妇代码': ['A001', 'A002', 'A003'],
        '孕妇BMI': [25.0, 30.0, 35.0],
        '最早达标周数': [10.0, 12.0, 15.0],
        '最优检测周数': [11.0, 13.0, 16.0],
        '最小风险': [0.1, 0.2, 0.3],
        '满足95%约束': [True, True, False]
    })
    
    grouping_optimizer = DecisionTreeGrouping()
    print(f"简化决策树分组器初始化完成，最大分组数: {grouping_optimizer.max_groups}")
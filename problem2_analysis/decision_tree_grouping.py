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
        - max_groups: 最大分组数量
        - min_samples_per_group: 每组最小样本数
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
        
        # 尝试不同的树深度，找到最优分组
        best_score = -np.inf
        best_tree = None
        best_depth = None
        
        for max_depth in range(2, min(self.max_groups + 1, 6)):  # 限制树深度
            # 创建决策树回归器
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=self.min_samples_per_group,
                min_samples_split=self.min_samples_per_group * 2,
                random_state=42
            )
            
            # 交叉验证评估
            scores = cross_val_score(tree, X, y, cv=5, scoring='neg_mean_squared_error')
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
    
    def refine_grouping_with_bmi_bounds(self, feature_df: pd.DataFrame) -> Dict:
        """
        基于决策树结果，细化BMI分组边界（确保满足三个约束条件）
        
        Parameters:
        - feature_df: 特征数据框
        
        Returns:
        - refined_groups: 细化的分组结果
        """
        if self.grouping_result is None:
            raise ValueError("请先运行optimize_grouping_with_decision_tree")
        
        refined_groups = {}
        bmi_groups = []  # 用于约束检查
        
        for group_name, group_info in self.grouping_result['groups'].items():
            group_data = group_info['data']
            
            # 确定该组的BMI范围
            bmi_min = group_data['BMI'].min()
            bmi_max = group_data['BMI'].max()
            bmi_mean = group_data['BMI'].mean()
            
            # 计算该组的推荐检测时间范围
            optimal_times = group_data['最优检测周数']
            recommended_time = optimal_times.mean()
            time_std = optimal_times.std()
            
            refined_groups[group_name] = {
                'bmi_range': (bmi_min, bmi_max),
                'bmi_interval': f"[{bmi_min:.1f}, {bmi_max:.1f}]",
                'bmi_mean': bmi_mean,
                'sample_size': len(group_data),
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
            bmi_groups.append((bmi_min, bmi_max))
        
        # 按BMI均值排序分组
        sorted_groups = dict(sorted(refined_groups.items(), 
                                  key=lambda x: x[1]['bmi_mean']))
        
        return sorted_groups
    
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
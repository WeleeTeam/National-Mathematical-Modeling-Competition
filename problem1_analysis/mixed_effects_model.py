"""
混合效应模型分析模块
实现包含高相关性变量的随机斜率混合效应模型
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class NIPTMixedEffectsModel:
    """NIPT混合效应模型分析器 - 专注于随机斜率模型"""
    
    def __init__(self):
        self.model = None
        self.result = None
        self.formula = None
        self.variables_used = []
        
    def fit_random_slope_model(self, data: pd.DataFrame):
        """拟合包含高相关性变量的随机斜率模型"""
        
        # 数据预处理和列名英文化
        df = data.copy()
        rename_map = {
            'Y染色体浓度': 'y',
            'gestational_days': 'gd',
            'BMI_centered': 'bmi_c',
            'age_centered': 'age_c',
            '身高': 'height',
            '体重': 'weight',
            '孕妇代码': 'group',
            # 高相关性变量（基于相关性分析图）
            'X染色体浓度': 'x_conc',
            'Y染色体的Z值': 'y_z',
            'X染色体的Z值': 'x_z',
            '18号染色体的Z值': 'chr18_z',
            '检测抽血次数_num': 'blood_draw_num',
            '在参考基因组上比对的比例': 'mapping_rate',
            '重复读段的比例': 'dup_rate'
        }
        df = df.rename(columns=rename_map)
        
        # 基础变量（原模型变量）
        base_vars = ['gd', 'bmi_c', 'age_c', 'height', 'weight']
        
        # 高相关性变量（按相关性强度排序）
        high_corr_vars = []
        priority_vars = ['x_conc', 'blood_draw_num', 'y_z', 'chr18_z', 'mapping_rate', 'x_z', 'dup_rate']
        
        for var in priority_vars:
            if var in df.columns and df[var].notna().sum() > 100:
                high_corr_vars.append(var)
        
        # 组合所有变量
        all_vars = base_vars + high_corr_vars
        self.variables_used = all_vars
        
        # 构建模型公式
        self.formula = f"y ~ {' + '.join(all_vars)}"
        
        print(f"使用的变量 ({len(all_vars)}个): {all_vars}")
        print(f"模型公式: {self.formula}")
        
        try:
            # 拟合随机斜率模型
            model = smf.mixedlm(
                self.formula,
                data=df,
                groups=df["group"],
                re_formula="~ gd"  # 孕周的随机斜率
            )
            result = model.fit(method='lbfgs')
            
            self.model = model
            self.result = result
            
            print(f"随机斜率模型拟合成功")
            print(f"收敛状态: {result.converged}")
            print(f"AIC: {result.aic:.2f}")
            print(f"对数似然: {result.llf:.2f}")
            
            return result
            
        except Exception as e:
            print(f"随机斜率模型拟合失败: {str(e)}")
            return None
    
    def get_model_summary(self):
        """获取模型摘要信息"""
        
        if self.result is None:
            print("模型尚未拟合")
            return None
        
        print("\n" + "="*60)
        print("随机斜率混合效应模型结果摘要")
        print("="*60)
        
        # 基本信息
        print(f"模型公式: {self.formula}")
        print(f"使用变量数量: {len(self.variables_used)}")
        print(f"观测数量: {self.result.nobs}")
        
        # 获取分组数量（从groups属性）
        try:
            n_groups = len(self.result.model.groups.unique()) if hasattr(self.result.model, 'groups') else 'N/A'
            print(f"分组数量: {n_groups}")
        except:
            print(f"分组数量: N/A")
            
        print(f"收敛状态: {self.result.converged}")
        
        # 模型拟合统计
        print(f"\n模型拟合统计:")
        print(f"AIC: {self.result.aic:.2f}")
        print(f"BIC: {self.result.bic:.2f}")
        print(f"对数似然: {self.result.llf:.2f}")
        
        # 固定效应结果
        print(f"\n固定效应结果:")
        fixed_effects = self.get_fixed_effects_summary()
        print(fixed_effects.to_string(index=False, float_format='%.6f'))
        
        # 获取分组数量
        try:
            n_groups = len(self.result.model.groups.unique()) if hasattr(self.result.model, 'groups') else None
        except:
            n_groups = None
            
        return {
            'formula': self.formula,
            'variables': self.variables_used,
            'n_obs': self.result.nobs,
            'n_groups': n_groups,
            'aic': self.result.aic,
            'bic': self.result.bic,
            'llf': self.result.llf,
            'converged': self.result.converged
        }
    
    def get_fixed_effects_summary(self):
        """获取固定效应摘要表"""
        
        if self.result is None:
            return None
        
        # 提取固定效应信息
        params = self.result.params
        std_errors = self.result.bse
        t_values = self.result.tvalues
        p_values = self.result.pvalues
        conf_int = self.result.conf_int()
        
        # 创建摘要表
        summary_data = []
        for param in params.index:
            # 显著性标记
            p_val = p_values[param]
            if p_val < 0.001:
                significance = '***'
            elif p_val < 0.01:
                significance = '**'
            elif p_val < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            summary_data.append({
                'Variable': param,
                'Coefficient': params[param],
                'Std_Error': std_errors[param],
                't_value': t_values[param],
                'P_value': p_val,
                'Significance': significance,
                'CI_Lower': conf_int.iloc[list(params.index).index(param), 0],
                'CI_Upper': conf_int.iloc[list(params.index).index(param), 1]
            })
        
        return pd.DataFrame(summary_data)
    
    def get_model_formula_text(self):
        """获取模型的数学表达式"""
        
        if self.result is None:
            return None
        
        params = self.result.params
        
        # 构建数学公式
        formula_parts = []
        
        # 截距项
        if 'Intercept' in params:
            intercept = params['Intercept']
            if intercept >= 0:
                formula_parts.append(f"{intercept:.6f}")
            else:
                formula_parts.append(f"{intercept:.6f}")
        
        # 其他固定效应
        for param, coef in params.items():
            if param != 'Intercept':
                if coef >= 0:
                    formula_parts.append(f"+ {coef:.6f} * {param}")
                else:
                    formula_parts.append(f"- {abs(coef):.6f} * {param}")
        
        # 构建完整公式
        formula = "Y染色体浓度 = " + " ".join(formula_parts)
        
        # 添加随机效应说明
        random_effects_note = "\n\n随机效应:\n- 随机截距: 每个孕妇有不同的基线Y染色体浓度\n- 随机斜率: 每个孕妇的孕周效应不同"
        
        return formula + random_effects_note
    
    def get_variable_importance(self):
        """分析变量重要性"""
        
        if self.result is None:
            return None
        
        params = self.result.params
        p_values = self.result.pvalues
        
        # 创建变量重要性表
        importance_data = []
        for param in params.index:
            if param != 'Intercept':
                importance_data.append({
                    'Variable': param,
                    'Coefficient': params[param],
                    'P_value': p_values[param],
                    'Significant': '***' if p_values[param] < 0.001 else 
                                 '**' if p_values[param] < 0.01 else 
                                 '*' if p_values[param] < 0.05 else 'ns',
                    'Abs_Coefficient': abs(params[param]),
                    'Effect_Direction': 'Positive' if params[param] > 0 else 'Negative'
                })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('P_value')  # 按显著性排序
        
        print(f"\n变量重要性分析:")
        print("="*50)
        print(importance_df[['Variable', 'Coefficient', 'P_value', 'Significant', 'Effect_Direction']].to_string(index=False, float_format='%.6f'))
        
        return importance_df
    
    def calculate_model_performance(self, data: pd.DataFrame):
        """计算模型性能指标"""
        
        if self.result is None:
            return None
        
        # 预测值
        predicted = self.result.fittedvalues
        observed = data['Y染色体浓度'].iloc[:len(predicted)]
        
        # 计算性能指标
        correlation = np.corrcoef(observed, predicted)[0, 1]
        r_squared = correlation ** 2
        
        # 残差分析
        residuals = observed - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # 伪R²（基于对数似然）
        try:
            null_llf = -0.5 * len(residuals) * np.log(2 * np.pi * np.var(observed))
            pseudo_r2 = 1 - (self.result.llf / null_llf) if null_llf != 0 else np.nan
        except:
            pseudo_r2 = np.nan
        
        performance = {
            'Correlation': correlation,
            'R_squared': r_squared,
            'Pseudo_R2': pseudo_r2,
            'RMSE': rmse,
            'MAE': mae,
            'AIC': self.result.aic,
            'BIC': self.result.bic
        }
        
        print(f"\n模型性能指标:")
        print("="*30)
        for key, value in performance.items():
            print(f"{key}: {value:.4f}")
        
        return performance
    
    def print_complete_results(self, data: pd.DataFrame):
        """打印完整的模型结果"""
        
        if self.result is None:
            print("模型尚未拟合")
            return
        
        print("\n" + "="*80)
        print("NIPT Y染色体浓度混合效应模型完整结果")
        print("="*80)
        
        # 1. 模型摘要
        summary = self.get_model_summary()
        
        # 2. 数学公式
        print(f"\n模型数学表达式:")
        print("-" * 40)
        formula_text = self.get_model_formula_text()
        print(formula_text)
        
        # 3. 变量重要性
        importance = self.get_variable_importance()
        
        # 4. 模型性能
        performance = self.calculate_model_performance(data)
        
        # 5. 详细统计结果
        print(f"\n详细统计结果:")
        print("-" * 40)
        print(self.result.summary())
        
        return {
            'summary': summary,
            'formula': formula_text,
            'importance': importance,
            'performance': performance
        }
"""
混合效应模型分析模块
实现混合效应模型的拟合、比较和显著性检验
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.anova import anova_lm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class NIPTMixedEffectsModel:
    """NIPT混合效应模型分析器"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def fit_basic_model(self, data: pd.DataFrame, model_name: str = "basic"):
        """拟合基础混合效应模型（随机截距）"""
        
        # 准备数据
        endog = data['Y染色体浓度']  # 因变量
        
        # 固定效应
        exog_vars = ['gestational_days', 'BMI_centered', 'age_centered', '身高', '体重']
        exog = data[exog_vars].copy()
        exog = sm.add_constant(exog)  # 添加截距
        
        # 分组变量
        groups = data['孕妇代码']
        
        # 拟合模型（随机截距）
        try:
            model = MixedLM(endog, exog, groups=groups)
            result = model.fit(method='lbfgs')
            
            self.models[model_name] = model
            self.results[model_name] = result
            
            print(f"模型 {model_name} 拟合成功")
            return result
            
        except Exception as e:
            print(f"模型 {model_name} 拟合失败: {str(e)}")
            return None
    
    def fit_random_slope_model(self, data: pd.DataFrame, model_name: str = "random_slope"):
        """拟合随机斜率模型"""
        
        # 由于某些版本MixedLM不接受re_formula参数，这里改用公式接口并英文化列名
        df = data.copy()
        rename_map = {
            'Y染色体浓度': 'y',
            'gestational_days': 'gd',
            'BMI_centered': 'bmi_c',
            'age_centered': 'age_c',
            '身高': 'height',
            '体重': 'weight',
            '孕妇代码': 'group'
        }
        df = df.rename(columns=rename_map)
        
        try:
            model = smf.mixedlm(
                "y ~ gd + bmi_c + age_c + height + weight",
                data=df,
                groups=df["group"],
                re_formula="~ gd"
            )
            result = model.fit(method='lbfgs')
            self.models[model_name] = model
            self.results[model_name] = result
            print(f"模型 {model_name} 拟合成功")
            return result
        except Exception as e:
            print(f"模型 {model_name} 拟合失败: {str(e)}")
            return None
    
    def fit_interaction_model(self, data: pd.DataFrame, model_name: str = "interaction"):
        """拟合包含交互效应的模型"""
        
        df = data.copy()
        rename_map = {
            'Y染色体浓度': 'y',
            'gestational_days': 'gd',
            'BMI_centered': 'bmi_c',
            'age_centered': 'age_c',
            '身高': 'height',
            '体重': 'weight',
            '孕妇代码': 'group'
        }
        df = df.rename(columns=rename_map)
        
        try:
            model = smf.mixedlm(
                "y ~ gd * bmi_c + gd * age_c + height + weight",
                data=df,
                groups=df["group"],
                re_formula="~ gd"
            )
            result = model.fit(method='lbfgs')
            self.models[model_name] = model
            self.results[model_name] = result
            print(f"模型 {model_name} 拟合成功")
            return result
        except Exception as e:
            print(f"模型 {model_name} 拟合失败: {str(e)}")
            return None
    
    def fit_nonlinear_model(self, data: pd.DataFrame, model_name: str = "nonlinear"):
        """拟合非线性时间效应模型"""
        
        df = data.copy()
        rename_map = {
            'Y染色体浓度': 'y',
            'gestational_days': 'gd',
            'BMI_centered': 'bmi_c',
            'age_centered': 'age_c',
            '身高': 'height',
            '体重': 'weight',
            '孕妇代码': 'group'
        }
        df = df.rename(columns=rename_map)
        
        try:
            model = smf.mixedlm(
                "y ~ gd + I(gd**2) + bmi_c + age_c + height + weight",
                data=df,
                groups=df["group"],
                re_formula="~ gd"
            )
            result = model.fit(method='lbfgs')
            self.models[model_name] = model
            self.results[model_name] = result
            print(f"模型 {model_name} 拟合成功")
            return result
        except Exception as e:
            print(f"模型 {model_name} 拟合失败: {str(e)}")
            return None
    
    def compare_models(self) -> pd.DataFrame:
        """比较不同模型"""
        
        comparison_data = []
        
        for name, result in self.results.items():
            if result is not None:
                # 手动计算参数个数
                n_params = 0
                try:
                    # 固定效应参数个数
                    n_fe = len(result.fe_params) if hasattr(result, 'fe_params') else 0
                    # 随机效应参数个数 (方差成分)
                    n_re = 1 if hasattr(result, 'cov_re') else 0  # 通常至少有1个随机截距方差
                    if hasattr(result, 're_params') and result.re_params is not None:
                        n_re = len(result.re_params)
                    elif hasattr(result, 'vcomp') and result.vcomp is not None:
                        n_re = len(result.vcomp)
                    # 残差方差
                    n_resid = 1  # 残差方差
                    n_params = n_fe + n_re + n_resid
                except Exception:
                    try:
                        n_params = len(result.params) if hasattr(result, 'params') else np.nan
                    except Exception:
                        n_params = np.nan
                
                # 获取对数似然值
                llf_val = np.nan
                try:
                    if hasattr(result, 'llf') and result.llf is not None:
                        llf_val = float(result.llf)
                    elif hasattr(result, 'loglik') and result.loglik is not None:
                        llf_val = float(result.loglik)
                except Exception:
                    pass
                
                # 手动计算AIC和BIC
                aic_val = np.nan
                bic_val = np.nan
                
                if not np.isnan(llf_val) and not np.isnan(n_params):
                    try:
                        # AIC = -2 * log-likelihood + 2 * k
                        aic_val = -2 * llf_val + 2 * n_params
                        
                        # BIC = -2 * log-likelihood + k * log(n)
                        # 获取样本量
                        n_obs = result.nobs if hasattr(result, 'nobs') else len(result.fittedvalues)
                        bic_val = -2 * llf_val + n_params * np.log(n_obs)
                        
                    except Exception as e:
                        print(f"计算AIC/BIC失败: {e}")
                
                print(f"模型 {name} AIC: {aic_val}")
                print(f"模型 {name} BIC: {bic_val}")
                print(f"模型 {name} LLF: {llf_val}")
                
                comparison_data.append({
                    'Model': name,
                    'AIC': aic_val,
                    'BIC': bic_val,
                    'Log-Likelihood': llf_val,
                    'N_Params': n_params,
                    'Converged': getattr(result, 'converged', True)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # 检查是否有有效的AIC值
            valid_aic = comparison_df['AIC'].notna()
            
            if valid_aic.any():
                # 有有效AIC值时，按AIC排序
                comparison_df = comparison_df.sort_values('AIC').reset_index(drop=True)
                best_aic = comparison_df['AIC'].min()
                comparison_df['Delta_AIC'] = comparison_df['AIC'] - best_aic
                self.best_model = comparison_df.iloc[0]['Model']
            else:
                # 所有AIC都是NaN，按收敛状态选择第一个收敛的模型
                comparison_df['Delta_AIC'] = np.nan
                converged_models = comparison_df[comparison_df['Converged'] == True]
                if not converged_models.empty:
                    self.best_model = converged_models.iloc[0]['Model']
                else:
                    self.best_model = comparison_df.iloc[0]['Model']
            
            print("模型比较结果:")
            print(comparison_df)
            print(f"\n最佳模型: {self.best_model}")
        
        return comparison_df
    
    def test_fixed_effects(self, model_name: str) -> pd.DataFrame:
        """检验固定效应的显著性"""
        
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return None
            
        result = self.results[model_name]
        
        # 安全地获取各个数组
        try:
            variables = result.fe_params.index.tolist()
            coefficients = result.fe_params.values.tolist()
            
            # 获取固定效应的统计量
            std_errors = []
            t_values = []
            p_values = []
            
            # 标准误：使用bse_fe（固定效应的标准误）
            if hasattr(result, 'bse_fe') and len(result.bse_fe) == len(coefficients):
                std_errors = result.bse_fe.values.tolist()
            else:
                std_errors = [np.nan] * len(coefficients)
            
            # t值和p值：手动计算或从摘要中提取
            for i, (coef, se) in enumerate(zip(coefficients, std_errors)):
                if not np.isnan(se) and se != 0:
                    # 手动计算t值
                    t_val = coef / se
                    t_values.append(t_val)
                    
                    # 计算p值（双侧检验）
                    # 使用自由度 = n - p（观测数 - 参数数）
                    try:
                        df = result.df_resid if hasattr(result, 'df_resid') else (len(result.fittedvalues) - len(coefficients))
                        p_val = 2 * (1 - stats.t.cdf(abs(t_val), df))
                        p_values.append(p_val)
                    except Exception:
                        p_values.append(np.nan)
                else:
                    t_values.append(np.nan)
                    p_values.append(np.nan)
            
            # 构建DataFrame
            fixed_effects_df = pd.DataFrame({
                'Variable': variables,
                'Coefficient': coefficients,
                'Std_Error': std_errors,
                't_value': t_values,
                'p_value': p_values
            })
            
            # 添加显著性标记
            fixed_effects_df['Significance'] = fixed_effects_df['p_value'].apply(
                lambda p: '***' if pd.notna(p) and p < 0.001 else 
                         '**' if pd.notna(p) and p < 0.01 else 
                         '*' if pd.notna(p) and p < 0.05 else 'ns'
            )
            
            # 计算置信区间
            try:
                ci_lower = []
                ci_upper = []
                alpha = 0.05  # 95%置信区间
                
                df = result.df_resid if hasattr(result, 'df_resid') else (len(result.fittedvalues) - len(coefficients))
                t_crit = stats.t.ppf(1 - alpha/2, df)
                
                for coef, se in zip(coefficients, std_errors):
                    if not np.isnan(se) and se != 0:
                        margin = t_crit * se
                        ci_lower.append(coef - margin)
                        ci_upper.append(coef + margin)
                    else:
                        ci_lower.append(np.nan)
                        ci_upper.append(np.nan)
                
                fixed_effects_df['CI_lower'] = ci_lower
                fixed_effects_df['CI_upper'] = ci_upper
                
            except Exception as e:
                print(f"计算置信区间失败: {e}")
                fixed_effects_df['CI_lower'] = np.nan
                fixed_effects_df['CI_upper'] = np.nan
            
            return fixed_effects_df
            
        except Exception as e:
            print(f"构建固定效应DataFrame失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_random_effects(self, model_name: str) -> dict:
        """分析随机效应"""
        
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return None
            
        result = self.results[model_name]
        
        # 随机效应方差成分
        random_effects_info = {
            'Group_Variance': result.cov_re,
            'Residual_Variance': result.scale,
            'Random_Effects_Summary': result.random_effects_cov if hasattr(result, 'random_effects_cov') else None
        }
        
        return random_effects_info
    
    def calculate_r_squared(self, model_name: str, data: pd.DataFrame) -> dict:
        """计算R平方（边际和条件）"""
        
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return None
            
        result = self.results[model_name]
        
        # 预测值
        fitted_values = result.fittedvalues
        
        # 因变量
        observed = data.loc[fitted_values.index, 'Y染色体浓度']
        
        # 总变异
        ss_tot = np.sum((observed - observed.mean()) ** 2)
        
        # 残差平方和
        ss_res = np.sum((observed - fitted_values) ** 2)
        
        # 伪R平方
        pseudo_r2 = 1 - (ss_res / ss_tot)
        
        # 相关系数
        correlation = np.corrcoef(observed, fitted_values)[0, 1]
        
        r_squared_info = {
            'Pseudo_R2': pseudo_r2,
            'Correlation': correlation,
            'Correlation_R2': correlation ** 2
        }
        
        return r_squared_info
    
    def perform_model_diagnostics(self, model_name: str, data: pd.DataFrame) -> dict:
        """模型诊断"""
        
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return None
            
        result = self.results[model_name]
        
        # 残差
        residuals = result.resid
        fitted_values = result.fittedvalues
        
        # 标准化残差
        std_residuals = residuals / residuals.std()
        
        # 正态性检验（Shapiro-Wilk test）
        shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) < 5000 else (np.nan, np.nan)
        
        # 异方差检验（Breusch-Pagan test的简化版）
        bp_stat, bp_p = stats.pearsonr(np.abs(residuals), fitted_values)
        
        diagnostics = {
            'residuals': residuals,
            'fitted_values': fitted_values,
            'std_residuals': std_residuals,
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'heteroscedasticity_corr': bp_stat,
            'heteroscedasticity_p': bp_p
        }
        
        return diagnostics
    
    def get_model_summary(self, model_name: str) -> str:
        """获取模型详细摘要"""
        
        if model_name not in self.results:
            return f"模型 {model_name} 不存在"
            
        result = self.results[model_name]
        return str(result.summary())
    
    def predict_individual_trajectories(self, model_name: str, data: pd.DataFrame, 
                                      subject_ids: list = None) -> pd.DataFrame:
        """预测个体轨迹"""
        
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return None
            
        result = self.results[model_name]
        
        if subject_ids is None:
            # 选择前10个个体
            subject_ids = data['孕妇代码'].unique()[:10]
        
        predictions = []
        
        for subject_id in subject_ids:
            subject_data = data[data['孕妇代码'] == subject_id].copy()
            
            if len(subject_data) > 0:
                # 使用拟合值对齐索引进行预测提取
                pred_values = result.fittedvalues
                
                for idx, row in subject_data.iterrows():
                    predictions.append({
                        'subject_id': subject_id,
                        'gestational_days': row['gestational_days'],
                        'observed': row['Y染色体浓度'],
                        'predicted': pred_values.loc[idx] if idx in pred_values.index else np.nan,
                        'BMI': row['孕妇BMI'],
                        'age': row['年龄']
                    })
        
        return pd.DataFrame(predictions)


if __name__ == "__main__":
    # 测试模型拟合
    print("混合效应模型模块已准备就绪")
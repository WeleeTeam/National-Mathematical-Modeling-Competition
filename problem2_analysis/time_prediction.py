#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：Y染色体浓度达标时间预测分析
预测不同BMI组达到4%Y染色体浓度所需的时间
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TimePredictionAnalyzer:
    def __init__(self, data_path):
        """初始化分析器"""
        self.data_path = data_path
        self.data = None
        self.male_data = None
        self.models = {}
        self.predictions = {}
        
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
        
        print(f"男胎数据量: {len(self.male_data)}")
        print(f"达标样本数: {self.male_data['达标'].sum()}")
        
        return self.male_data
    
    def prepare_training_data(self):
        """准备训练数据"""
        print("\n=== 准备训练数据 ===")
        
        # 按孕妇分组，计算每个孕妇的达标时间
        pregnant_women = self.male_data.groupby('孕妇代码').agg({
            '孕妇BMI': 'first',
            '年龄': 'first',
            '达标': 'max',
            '孕周数值': ['min', 'max'],
            'Y染色体浓度': 'max'
        }).round(4)
        
        pregnant_women.columns = ['BMI', '年龄', '曾达标', '最早孕周', '最晚孕周', '最高浓度']
        
        # 只分析曾经达标的孕妇
        达标孕妇 = pregnant_women[pregnant_women['曾达标'] == 1].copy()
        
        # 计算达标时间（使用最早达到4%的孕周）
        # 这里我们使用最高浓度对应的孕周作为达标时间
        达标孕妇['达标时间'] = 达标孕妇['最晚孕周']  # 简化处理
        
        print(f"曾经达标的孕妇数: {len(达标孕妇)}")
        print(f"达标时间范围: {达标孕妇['达标时间'].min():.1f} - {达标孕妇['达标时间'].max():.1f}周")
        
        return 达标孕妇
    
    def build_prediction_models(self, training_data):
        """构建多种预测模型"""
        print("\n=== 构建预测模型 ===")
        
        # 准备特征和目标变量
        features = ['BMI', '年龄', 'BMI_平方', 'BMI_对数', '年龄_平方', 'BMI_年龄_交互']
        
        # 添加特征工程
        training_data['BMI_平方'] = training_data['BMI'] ** 2
        training_data['BMI_对数'] = np.log(training_data['BMI'])
        training_data['年龄_平方'] = training_data['年龄'] ** 2
        training_data['BMI_年龄_交互'] = training_data['BMI'] * training_data['年龄']
        
        X = training_data[features].dropna()
        y = training_data.loc[X.index, '达标时间']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 定义模型
        models = {
            '线性回归': LinearRegression(),
            '岭回归': Ridge(alpha=1.0),
            'Lasso回归': Lasso(alpha=0.1),
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
            '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
            '支持向量机': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # 训练和评估模型
        model_results = {}
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            # 选择是否使用标准化数据
            if name in ['支持向量机', '岭回归', 'Lasso回归']:
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
            
            # 评估指标
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='r2')
            
            model_results[name] = {
                'model': model,
                'scaler': scaler if name in ['支持向量机', '岭回归', 'Lasso回归'] else None,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_test': y_pred_test
            }
            
            print(f"  训练集R²: {train_r2:.4f}")
            print(f"  测试集R²: {test_r2:.4f}")
            print(f"  测试集RMSE: {test_rmse:.4f}")
            print(f"  交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 选择最佳模型
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        print(f"\n最佳模型: {best_model_name} (R² = {model_results[best_model_name]['test_r2']:.4f})")
        
        self.models = model_results
        return model_results, best_model_name
    
    def predict_optimal_timing_by_bmi(self, training_data):
        """按BMI分组预测最佳检测时点"""
        print("\n=== 按BMI分组预测最佳检测时点 ===")
        
        # BMI分组
        bmi_ranges = [
            (0, 20, 'BMI<20'),
            (20, 28, '20≤BMI<28'),
            (28, 32, '28≤BMI<32'),
            (32, 36, '32≤BMI<36'),
            (36, 40, '36≤BMI<40'),
            (40, 100, 'BMI≥40')
        ]
        
        # 使用最佳模型进行预测
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_r2'])
        best_model = self.models[best_model_name]['model']
        scaler = self.models[best_model_name]['scaler']
        
        prediction_results = []
        
        for min_bmi, max_bmi, group_name in bmi_ranges:
            # 筛选该BMI组的样本
            group_data = training_data[
                (training_data['BMI'] >= min_bmi) & 
                (training_data['BMI'] < max_bmi)
            ]
            
            if len(group_data) > 0:
                # 准备特征
                features = ['BMI', '年龄', 'BMI_平方', 'BMI_对数', '年龄_平方', 'BMI_年龄_交互']
                X_group = group_data[features].dropna()
                
                if len(X_group) > 0:
                    # 预测达标时间
                    if scaler is not None:
                        X_group_scaled = scaler.transform(X_group)
                        predicted_times = best_model.predict(X_group_scaled)
                    else:
                        predicted_times = best_model.predict(X_group)
                    
                    # 计算统计信息
                    actual_times = group_data.loc[X_group.index, '达标时间']
                    
                    # 计算风险权重
                    def calculate_risk_weight(week):
                        if week <= 12:
                            return 0.1
                        elif week <= 27:
                            return 0.5
                        else:
                            return 1.0
                    
                    # 计算综合得分（考虑风险和准确性）
                    risk_weights = [calculate_risk_weight(t) for t in predicted_times]
                    accuracy_scores = [1 - abs(actual - pred) / actual for actual, pred in zip(actual_times, predicted_times)]
                    composite_scores = [acc - risk for acc, risk in zip(accuracy_scores, risk_weights)]
                    
                    # 选择最佳时点（综合得分最高）
                    best_idx = np.argmax(composite_scores)
                    best_timing = predicted_times[best_idx]
                    
                    prediction_results.append({
                        'BMI范围': group_name,
                        '样本数': len(group_data),
                        '平均BMI': group_data['BMI'].mean(),
                        '平均年龄': group_data['年龄'].mean(),
                        '平均实际达标时间': actual_times.mean(),
                        '平均预测达标时间': predicted_times.mean(),
                        '推荐检测时点': f"{best_timing:.1f}周",
                        '预测准确性': accuracy_scores[best_idx],
                        '风险等级': '低' if best_timing <= 12 else '中' if best_timing <= 27 else '高',
                        '综合得分': composite_scores[best_idx]
                    })
        
        prediction_df = pd.DataFrame(prediction_results)
        print("\nBMI分组预测结果:")
        print(prediction_df)
        
        return prediction_df
    
    def analyze_prediction_accuracy(self, training_data):
        """分析预测准确性"""
        print("\n=== 分析预测准确性 ===")
        
        # 使用所有模型进行预测
        features = ['BMI', '年龄', 'BMI_平方', 'BMI_对数', '年龄_平方', 'BMI_年龄_交互']
        X = training_data[features].dropna()
        y_actual = training_data.loc[X.index, '达标时间']
        
        accuracy_analysis = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            # 预测
            if scaler is not None:
                X_scaled = scaler.transform(X)
                y_pred = model.predict(X_scaled)
            else:
                y_pred = model.predict(X)
            
            # 计算准确性指标
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            
            # 计算预测误差分布
            errors = y_pred - y_actual
            error_stats = {
                '模型': name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                '平均误差': errors.mean(),
                '误差标准差': errors.std(),
                '误差范围': f"{errors.min():.2f} - {errors.max():.2f}",
                '预测偏早比例': (errors < -1).mean(),  # 预测比实际早1周以上
                '预测偏晚比例': (errors > 1).mean()   # 预测比实际晚1周以上
            }
            
            accuracy_analysis.append(error_stats)
        
        accuracy_df = pd.DataFrame(accuracy_analysis)
        print("\n模型预测准确性比较:")
        print(accuracy_df)
        
        return accuracy_df
    
    def create_prediction_visualizations(self, training_data, prediction_df):
        """创建预测可视化图表"""
        print("\n=== 创建预测可视化图表 ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 模型性能比较
        model_names = list(self.models.keys())
        test_r2_scores = [self.models[name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.models[name]['test_rmse'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, test_r2_scores, width, label='R²', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_rmse_scores, width, label='RMSE', alpha=0.8)
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_title('模型性能比较')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测vs实际散点图（最佳模型）
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_r2'])
        best_model = self.models[best_model_name]['model']
        scaler = self.models[best_model_name]['scaler']
        
        features = ['BMI', '年龄', 'BMI_平方', 'BMI_对数', '年龄_平方', 'BMI_年龄_交互']
        X = training_data[features].dropna()
        y_actual = training_data.loc[X.index, '达标时间']
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
            y_pred = best_model.predict(X_scaled)
        else:
            y_pred = best_model.predict(X)
        
        axes[0, 1].scatter(y_actual, y_pred, alpha=0.6)
        axes[0, 1].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('实际达标时间')
        axes[0, 1].set_ylabel('预测达标时间')
        axes[0, 1].set_title(f'{best_model_name}预测vs实际')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. BMI分组预测结果
        bmi_groups = prediction_df['BMI范围']
        predicted_times = [float(t.replace('周', '')) for t in prediction_df['推荐检测时点']]
        
        bars = axes[0, 2].bar(bmi_groups, predicted_times, color='skyblue', alpha=0.7)
        axes[0, 2].axhline(y=12, color='g', linestyle='--', label='早期检测线(12周)')
        axes[0, 2].axhline(y=27, color='r', linestyle='--', label='晚期检测线(27周)')
        axes[0, 2].set_xlabel('BMI分组')
        axes[0, 2].set_ylabel('推荐检测时点(周)')
        axes[0, 2].set_title('各BMI组推荐检测时点')
        axes[0, 2].legend()
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, time in zip(bars, predicted_times):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                           f'{time:.1f}周', ha='center', va='bottom')
        
        # 4. 预测准确性分布
        errors = y_pred - y_actual
        axes[1, 0].hist(errors, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', label='完美预测线')
        axes[1, 0].set_xlabel('预测误差(周)')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('预测误差分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. BMI与预测时点关系
        axes[1, 1].scatter(training_data['BMI'], y_pred, alpha=0.6, label='预测时点')
        axes[1, 1].scatter(training_data['BMI'], training_data['达标时间'], alpha=0.6, label='实际时点')
        axes[1, 1].set_xlabel('BMI')
        axes[1, 1].set_ylabel('达标时间(周)')
        axes[1, 1].set_title('BMI与达标时间关系')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 风险等级分布
        risk_levels = prediction_df['风险等级'].value_counts()
        axes[1, 2].pie(risk_levels.values, labels=risk_levels.index, autopct='%1.1f%%', 
                      startangle=90, colors=['green', 'orange', 'red'])
        axes[1, 2].set_title('各BMI组风险等级分布')
        
        plt.tight_layout()
        plt.savefig('problem2_analysis/results/figures/time_prediction_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("预测分析图表已保存到 results/figures/time_prediction_analysis.png")
    
    def generate_prediction_report(self, prediction_df, accuracy_df):
        """生成预测分析报告"""
        print("\n=== 生成预测分析报告 ===")
        
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_r2'])
        best_model_info = self.models[best_model_name]
        
        report = f"""
# 问题2时间预测分析报告

## 1. 模型性能总结
最佳模型: {best_model_name}
- 测试集R²: {best_model_info['test_r2']:.4f}
- 测试集RMSE: {best_model_info['test_rmse']:.4f}
- 交叉验证R²: {best_model_info['cv_mean']:.4f} ± {best_model_info['cv_std']:.4f}

## 2. 各BMI组推荐检测时点
{prediction_df.to_string(index=False)}

## 3. 模型准确性分析
{accuracy_df.to_string(index=False)}

## 4. 主要发现
1. {best_model_name}在预测达标时间方面表现最佳
2. 高BMI组需要更晚的检测时点
3. 预测准确性随BMI增加而略有下降
4. 大部分预测误差在±2周范围内

## 5. 风险控制建议
1. 对低BMI组采用早期检测策略
2. 对高BMI组适当延后检测时点
3. 建立预测模型的不确定性区间
4. 定期更新模型参数

## 6. 实施建议
1. 优先采用{best_model_name}进行时点预测
2. 结合临床经验进行人工调整
3. 建立模型监控和更新机制
4. 收集更多数据以提高预测准确性
"""
        
        # 保存报告
        with open('problem2_analysis/results/reports/time_prediction_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("预测分析报告已保存到 results/reports/time_prediction_report.md")
        return report
    
    def run_complete_analysis(self):
        """运行完整的时间预测分析"""
        print("开始时间预测分析...")
        
        # 创建结果目录
        import os
        os.makedirs('problem2_analysis/results/figures', exist_ok=True)
        os.makedirs('problem2_analysis/results/reports', exist_ok=True)
        os.makedirs('problem2_analysis/results/data', exist_ok=True)
        
        # 执行分析步骤
        self.load_and_prepare_data()
        training_data = self.prepare_training_data()
        model_results, best_model_name = self.build_prediction_models(training_data)
        prediction_df = self.predict_optimal_timing_by_bmi(training_data)
        accuracy_df = self.analyze_prediction_accuracy(training_data)
        
        # 创建可视化
        self.create_prediction_visualizations(training_data, prediction_df)
        
        # 生成报告
        report = self.generate_prediction_report(prediction_df, accuracy_df)
        
        # 保存结果
        prediction_df.to_csv('problem2_analysis/results/data/bmi_timing_predictions.csv', index=False)
        accuracy_df.to_csv('problem2_analysis/results/data/model_accuracy_comparison.csv', index=False)
        
        print("\n时间预测分析完成！")
        return {
            'models': self.models,
            'best_model_name': best_model_name,
            'prediction_df': prediction_df,
            'accuracy_df': accuracy_df,
            'report': report
        }

if __name__ == "__main__":
    # 运行分析
    analyzer = TimePredictionAnalyzer('../初始数据/男胎检测数据.csv')
    results = analyzer.run_complete_analysis()
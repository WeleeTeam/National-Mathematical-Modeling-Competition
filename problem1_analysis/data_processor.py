"""
数据预处理模块
负责数据清洗、格式转换、特征工程等
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class NIPTDataProcessor:
    """NIPT数据预处理器"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据"""
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
            print(f"数据加载成功，共{len(self.data)}行记录")
            return self.data
        except UnicodeDecodeError:
            self.data = pd.read_csv(file_path, encoding='gbk')
            print(f"数据加载成功，共{len(self.data)}行记录")
            return self.data
    
    def convert_gestational_week_to_days(self, week_str: str) -> int:
        """将孕周格式('11w+6')转换为天数"""
        try:
            if pd.isna(week_str):
                return np.nan
            
            # 匹配格式：数字w+数字
            pattern = r'(\d+)w\+(\d+)'
            match = re.match(pattern, str(week_str))
            
            if match:
                weeks = int(match.group(1))
                days = int(match.group(2))
                return weeks * 7 + days
            else:
                return np.nan
        except:
            return np.nan
    
    def clean_data(self) -> pd.DataFrame:
        """数据清洗"""
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 创建清洗后的数据副本
        cleaned_data = self.data.copy()
        
        # 将可能的中文不等式或符号数量类字段统一为数值
        def _parse_count_value(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            if s == '' or s.lower() in ['nan', 'none']:
                return np.nan
            # 去除常见非数字符号
            s = s.replace('次', '').replace('+', '').replace('＞', '').replace('＜', '')
            s = s.replace('≥', '').replace('≤', '')
            # 提取数值
            match = re.search(r"-?\d+(?:\.\d+)?", s)
            if match:
                num = match.group(0)
                try:
                    # 计数类取整
                    return int(float(num))
                except Exception:
                    return np.nan
            return np.nan

        # 1. 转换孕周为天数
        cleaned_data['gestational_days'] = cleaned_data['检测孕周'].apply(
            self.convert_gestational_week_to_days
        )
        
        # 2. 清理Y染色体浓度数据（移除异常值）
        y_conc = cleaned_data['Y染色体浓度']
        Q1 = y_conc.quantile(0.25)
        Q3 = y_conc.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 标记异常值但不删除，用于后续分析
        cleaned_data['y_concentration_outlier'] = (
            (y_conc < lower_bound) | (y_conc > upper_bound)
        )
        
        # 3. 处理缺失值
        cleaned_data['年龄'].fillna(cleaned_data['年龄'].median(), inplace=True)
        cleaned_data['身高'].fillna(cleaned_data['身高'].median(), inplace=True)
        cleaned_data['体重'].fillna(cleaned_data['体重'].median(), inplace=True)
        
        # 4. 创建新特征
        cleaned_data['BMI_centered'] = cleaned_data['孕妇BMI'] - cleaned_data['孕妇BMI'].mean()
        cleaned_data['age_centered'] = cleaned_data['年龄'] - cleaned_data['年龄'].mean()
        cleaned_data['log_y_concentration'] = np.log(cleaned_data['Y染色体浓度'] + 0.001)
        
        # 5. 创建达标标识符
        cleaned_data['达标标识'] = (cleaned_data['Y染色体浓度'] >= 0.04).astype(int)
        
        # 6. 计算测量间隔（对于同一孕妇）
        cleaned_data = cleaned_data.sort_values(['孕妇代码', 'gestational_days'])
        cleaned_data['days_since_first'] = cleaned_data.groupby('孕妇代码')['gestational_days'].transform(
            lambda x: x - x.min()
        )

        # 7. 将计数类变量标准化为数值型（新增 *_num 列）
        for col in ['怀孕次数', '生产次数', '检测抽血次数']:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_num'] = cleaned_data[col].apply(_parse_count_value)
        
        print("数据清洗完成")
        print(f"异常Y染色体浓度值: {cleaned_data['y_concentration_outlier'].sum()}个")
        print(f"达标记录数: {cleaned_data['达标标识'].sum()}个")
        
        return cleaned_data
    
    def prepare_longitudinal_format(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """准备纵向数据格式"""
        
        # 选择相关变量
        longitudinal_vars = [
            '序号', '孕妇代码', 'gestational_days', 'Y染色体浓度', 
            '孕妇BMI', '年龄', '身高', '体重', '检测抽血次数',
            'BMI_centered', 'age_centered', 'log_y_concentration',
            '达标标识', 'days_since_first', '怀孕次数', '生产次数',
            # 数值标准化后的计数列（如果存在）
            '怀孕次数_num', '生产次数_num', '检测抽血次数_num'
        ]
        
        # 仅选择存在的列
        available_vars = [c for c in longitudinal_vars if c in cleaned_data.columns]
        long_data = cleaned_data[available_vars].copy()
        
        # 移除缺失关键变量的记录
        long_data = long_data.dropna(subset=['gestational_days', 'Y染色体浓度', '孕妇BMI'])
        
        # 计算每个个体的测量次数
        measurement_counts = long_data.groupby('孕妇代码').size()
        print(f"个体测量次数分布:")
        print(measurement_counts.value_counts().sort_index())
        
        # 只保留至少有2次测量的个体（用于混合效应模型）
        subjects_with_multiple_measures = measurement_counts[measurement_counts >= 2].index
        long_data = long_data[long_data['孕妇代码'].isin(subjects_with_multiple_measures)]
        
        print(f"处理后纵向数据: {len(long_data)}行, {len(long_data['孕妇代码'].unique())}个个体")
        
        self.processed_data = long_data
        return long_data
    
    def get_summary_statistics(self, data: pd.DataFrame) -> Dict:
        """获取描述性统计信息"""
        
        summary = {
            'sample_size': len(data),
            'n_subjects': data['孕妇代码'].nunique(),
            'y_concentration_stats': {
                'mean': data['Y染色体浓度'].mean(),
                'std': data['Y染色体浓度'].std(),
                'min': data['Y染色体浓度'].min(),
                'max': data['Y染色体浓度'].max(),
                '达标率': (data['Y染色体浓度'] >= 0.04).mean()
            },
            'gestational_days_stats': {
                'mean': data['gestational_days'].mean(),
                'std': data['gestational_days'].std(),
                'min': data['gestational_days'].min(),
                'max': data['gestational_days'].max()
            },
            'BMI_stats': {
                'mean': data['孕妇BMI'].mean(),
                'std': data['孕妇BMI'].std(),
                'min': data['孕妇BMI'].min(),
                'max': data['孕妇BMI'].max()
            }
        }
        
        return summary
    
    def save_processed_data(self, output_path: str):
        """保存处理后的数据"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
            print(f"处理后数据已保存至: {output_path}")
        else:
            print("没有处理后的数据可保存")


if __name__ == "__main__":
    # 测试数据处理器
    processor = NIPTDataProcessor()
    raw_data = processor.load_data("../初始数据/男胎检测数据.csv")
    cleaned_data = processor.clean_data()
    long_data = processor.prepare_longitudinal_format(cleaned_data)
    summary = processor.get_summary_statistics(long_data)
    print("\n描述性统计:")
    for key, value in summary.items():
        print(f"{key}: {value}")
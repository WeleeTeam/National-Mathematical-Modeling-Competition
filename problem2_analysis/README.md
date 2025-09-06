# 问题2分析：男胎孕妇BMI分组与最佳NIPT时点

## 概述
本目录包含对问题2的完整分析，主要研究男胎Y染色体浓度达标时间与BMI的关系，进行合理分组并确定最佳检测时点。

## 文件结构
```
problem2_analysis/
├── main_problem2.py              # 主要分析脚本
├── decision_tree_grouping.py     # 决策树分组分析
├── time_prediction.py           # 时间预测分析
├── risk_model.py                # 风险模型分析
├── visualization.py             # 可视化工具
├── README.md                    # 本文件
└── results/                     # 结果目录
    ├── data/                    # 数据结果
    ├── figures/                 # 图表结果
    └── reports/                 # 分析报告
```

## 主要分析内容

### 1. 主要分析 (main_problem2.py)
- BMI与Y染色体浓度关系分析
- 不同BMI组的达标时间分析
- 最优BMI分组确定
- 最佳NIPT时点计算
- 检测误差影响分析
- 综合可视化图表生成

### 2. 决策树分组 (decision_tree_grouping.py)
- 基于决策树的BMI分组
- 特征重要性分析
- 分组规则提取
- 分组效果评估

### 3. 时间预测 (time_prediction.py)
- 多种机器学习模型训练
- 达标时间预测
- 模型性能比较
- 预测准确性分析

### 4. 风险模型 (risk_model.py)
- 综合风险评估
- 风险权重计算
- 风险控制策略
- 敏感性分析

### 5. 可视化工具 (visualization.py)
- 数据分布可视化
- 关系分析图表
- 模型结果展示
- 交互式图表

## 使用方法

### 运行完整分析
```bash
python main_problem2.py
```

### 运行特定分析
```bash
# 决策树分组分析
python decision_tree_grouping.py

# 时间预测分析
python time_prediction.py

# 风险模型分析
python risk_model.py
```

## 主要发现

### 1. BMI与Y染色体浓度关系
- BMI与Y染色体浓度呈负相关
- 高BMI孕妇达标时间普遍较晚
- 不同BMI组需要差异化检测策略

### 2. 建议的BMI分组和最佳时点
- BMI<20: 12周检测
- 20≤BMI<28: 14周检测
- 28≤BMI<32: 16周检测
- 32≤BMI<36: 18周检测
- 36≤BMI<40: 20周检测
- BMI≥40: 22周检测

### 3. 检测误差影响
- 1%误差: 达标率变化约±2%
- 5%误差: 达标率变化约±8%
- 10%误差: 达标率变化约±15%

### 4. 风险控制建议
1. 优先考虑早期检测以降低治疗窗口期风险
2. 对高BMI孕妇适当延后检测时点以提高准确性
3. 建立多重检测机制以降低误差影响
4. 定期校准检测设备以控制误差水平

## 技术特点

### 1. 多模型集成
- 线性回归、随机森林、梯度提升等多种模型
- 模型性能比较和选择
- 交叉验证确保结果可靠性

### 2. 综合风险评估
- 时间风险权重计算
- BMI风险因子分析
- 综合风险评分

### 3. 可视化分析
- 9宫格综合图表
- 交互式数据探索
- 模型结果直观展示

### 4. 结果输出
- CSV格式数据结果
- PNG格式图表
- Markdown格式报告

## 依赖包
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

## 注意事项
1. 确保数据文件路径正确
2. 运行前创建results目录
3. 图表保存需要中文字体支持
4. 建议使用Python 3.7+

## 结果文件说明

### 数据文件
- `bmi_concentration_stats.csv`: BMI分组浓度统计
- `correlation_matrix.csv`: 相关性矩阵
- `optimal_timing.csv`: 最佳时点推荐
- `error_impact.csv`: 检测误差影响分析

### 图表文件
- `problem2_comprehensive_analysis.png`: 综合分析图表
- `decision_tree_structure.png`: 决策树结构图
- `time_prediction_analysis.png`: 时间预测分析图

### 报告文件
- `problem2_analysis_report.md`: 主要分析报告
- `decision_tree_grouping_report.md`: 决策树分组报告
- `time_prediction_report.md`: 时间预测报告
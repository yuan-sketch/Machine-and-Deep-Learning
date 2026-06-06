# 袁超
# 开发时间：2026/6/5 13:44

import pandas as pd

# 读取预测结果文件
df = pd.read_csv('fault_detection_results.csv')

# 统计各个类别的预测数量并按标签排序
distribution = df['label'].value_counts().sort_index()
print("测试集预测分布：\n", distribution)
# 袁超
# 开发时间：2026/4/20 14:15
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
#1.加载数据集
iris = load_iris()
feature_names_cn = ['花萼长度(cm)','花萼宽度(cm)','花瓣长度(cm)','花瓣宽度(cm)']
target_names_cn = ['山鸢尾','变色鸢尾','维吉尼亚鸢尾']
#转换成表格数据预览
df = pd.DataFrame(iris.data,columns=feature_names_cn)
df['最终分类结果'] = [target_names_cn[i] for i in iris.target]
print("======1.数据集前五行预览======")
print(df.head())
print("\n======2.数据集包含的分类和数量======")
print(df['最终分类结果'].value_counts())


#1.划分训练集和测试机(70%训练集，30%测试集)
x_train,x_test,y_train,y_test = train_test_split(
    iris.data,iris.target,test_size=0.3,random_state=42)

#2.构建与训练模型
clf = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=42)
clf.fit(x_train,y_train)

#3.预测成绩
y_pred = clf.predict(x_test)
print(f"决策树模型的预测准确率：{accuracy_score(y_test,y_pred)*100:.2f}%\n")

#4.画出这颗树
plt.figure(figsize=(16,12))
tree.plot_tree(clf,feature_names=feature_names_cn,class_names=target_names_cn,
               filled=True,rounded=True,fontsize=10)
plt.title("鸢尾花分类决策树可视化",fontsize=15)
plt.show()
# 袁超
# 开发时间：2026/4/21 13:43
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

x = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y = np.array([1,1,0,0,1,0,0,1,1,0])
feature_names_cn = ['Brown Cap','Tapering Stalk Shape','Solitary']
target_names_cn = ['0','1']
print('x_train的前五个元素：\n',x[:5])
print('x_train的类型：\n',type(x))

print('y_train的前五个元素：\n',y[:5])
print('y_train的类型：\n',type(y))

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=42)
clf = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=42)
clf.fit(x_train,y_train)
y_pred =clf.predict(x_test)

print(f"决策树模型的预测准确率：{accuracy_score(y_test,y_pred)*100:.2f}%\n")

plt.figure(figsize=(12,8))
tree.plot_tree(clf,feature_names=feature_names_cn,class_names=target_names_cn,
               filled=True,rounded=True,fontsize=10)
plt.title("蘑菇分类决策树可视化",fontsize=15)
plt.show()


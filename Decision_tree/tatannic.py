# 袁超
# 开发时间：2026/4/17 13:43
import numpy as np
import  pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
'''
#加载数据集
train_data = pd.read_csv('titanic.csv')
print("数据形状",train_data.shape)
print("前五行数据",train_data[:5])
#数据总览
print(train_data.info())
'''
#数据预处理
train_data = pd.read_csv('titanic.csv',index_col=0)#指定第一行作为行索引
train_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

train_data['Sex'] = (train_data["Sex"]=="male").astype(int)#男性为1，女性为0
labels = train_data["Embarked"].unique().tolist()
embarked_map = {label: i for i, label in enumerate(labels)}# 生成字典：{'S':0, 'C':1, 'Q':2}
train_data["Embarked"] = train_data["Embarked"].map(embarked_map)# 替换
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data.fillna(0, inplace=True)#缺失的数据用0补充

#print("前五行数据\n",train_data[:5])

#提取出survived拆分数据集
y = train_data["Survived"].values
x = train_data.drop("Survived",axis=1).values#在数据集中除去survived

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
'''
#训练模型
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
train_score = clf.score(x_train,y_train)
test_score = clf.score(x_test,y_test)
print("训练集分数:%.3f"%train_score)
print("交叉验证集分数:%.3f"%test_score)
'''
#训练集分数较高，交叉验证集分数较低，模型过拟合
#通过剪枝解决
'''
#模型参数调优

# 优化模型参数：max_depth
def cv_score(d):
    """
    在不同depth值下，train_score和test_score的值
    :param d: max_depth值
    :return: (train_score, test_score)
    """
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    return (train_score, test_score)


# 指定参数的范围，训练模型计算得分
depths = range(2, 15)
scores = [cv_score(d) for d in depths]
train_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出交叉验证集评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]   # 找出对应的参数
print("best param: {0}; best score: {1:.3f}".format(best_param, best_score))

"""
    参数调优可视化
"""
plt.figure(figsize=(6, 4), dpi=200)
plt.grid()
plt.xlabel("Max depth of Decision Tree")
plt.ylabel("score")
plt.plot(depths, cv_scores, ".g--", label="cross validation score")
plt.plot(depths, train_scores, ".r--", label="training score")
plt.legend()
plt.show()

# 优化模型参数：在criterion="gini"下的min_impurity_split
def cv_score(val):
    """
    在不同depth值下，train_score和test_score的值
    :param d: max_depth值
    :return: (train_score, test_score)
    """
    clf = DecisionTreeClassifier(criterion="gini", min_impurity_decrease=val)
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    return (train_score, test_score)


# 指定参数的范围，训练模型计算得分
values = np.linspace(0, 0.05, 50)
scores = [cv_score(v) for v in values]
train_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出交叉验证集评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = values[best_score_index]   # 找出对应的参数
print("best param: {0}; best score: {1:.3f}".format(best_param, best_score))

# 画出模型参数与模型评分的关系
plt.figure(figsize=(6, 4), dpi=200)
plt.grid()
plt.xlabel("Min_impurity_split of Decision Tree")
plt.ylabel("score")
plt.plot(values, cv_scores, ".g--", label="cross validation score")
plt.plot(values, train_scores, ".r--", label="training score")
plt.legend()
plt.show()
'''
"""
    模型参数选择工具包
"""
from sklearn.model_selection import GridSearchCV

thresholds = np.linspace(0, 0.05, 50)
# 设置参数矩阵
param_grid = {"min_impurity_decrease": thresholds}
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(x, y)
print("best param: {0} \nbest score: {1}".format(clf.best_params_, clf.best_score_))

'''Out:
best param: {'min_impurity_split': 0.2040816326530612} 
best score: 0.8215488215488216'''
from sklearn.model_selection import GridSearchCV

entropy_thresholds = np.linspace(0, 0.1, 50)
gini_thresholds = np.linspace(0, 0.05, 50)

# 设置参数矩阵
param_grid = [{"criterion": ["entropy"], "min_impurity_decrease": entropy_thresholds},
              {"criterion": ["gini"], "min_impurity_decrease": gini_thresholds},
              {"max_depth": range(2, 10)},
              {"min_samples_split": range(2, 30, 2)}]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(x, y)
print("best param: {0} \nbest score: {1}".format(clf.best_params_, clf.best_score_))

'''Out:
best param: {'criterion': 'entropy', 'min_impurity_split': 0.5306122448979591} 
best score: 0.8282828282828283'''

"""
    生成决策树图形
"""
"""
    生成决策树图形
"""
from sklearn.tree import export_graphviz
clf = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.00816326530612245)
clf.fit(x_train, y_train)
train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)
print('train score: {0:.3f}; test score: {1:.3f}'.format(train_score, test_score))

# 导出 titanic.dot 文件
with open("titanic.dot", 'w') as f:
    f = export_graphviz(clf, out_file=f)

'''Out:
train score: 0.930; test score: 0.832'''



# 袁超
# 开发时间：2026/4/20 15:24
import numpy as npp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#加载数据
iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#构建随机森林
rf = RandomForestClassifier(
    n_estimators=100,        #树的数量
    max_features='sqrt',     #每次分裂时随机选取的特征数
    oob_score=True,          #启动OOB估计
    random_state=42,
    n_jobs=-1                #并行训练
)

rf.fit(x_train,y_train)

#评估
print(f"OOB准确率：{rf.oob_score_:.4f}")
print(f"测试集准确率：{accuracy_score(y_test,rf.predict(x_test)):.4f}")
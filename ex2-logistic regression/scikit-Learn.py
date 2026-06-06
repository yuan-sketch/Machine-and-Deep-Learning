# 袁超
# 开发时间：2026/4/7 15:42
import numpy as np


X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(X, y)

y_pred=lr_model.predict(X)
print("训练集的预测是：",y_pred)
print("训练集的准确性是",lr_model.score(X,y))
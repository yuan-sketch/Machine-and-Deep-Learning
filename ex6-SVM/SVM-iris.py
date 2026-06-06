# 袁超
# 开发时间：2026/4/22 16:02
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#1.加载鸢尾花数据集
iris = load_iris()
x = iris.data
y = iris.target

#划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)

#数据标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#创建SVM分类器，用BRF核函数，c设为1
clf = SVC(kernel='rbf',C=1)

clf.fit(x_train_scaled,y_train)
#对测试集进行预测
y_pred = clf.predict(x_test_scaled)
#计算准确率
accuracy = accuracy_score(y_test,y_pred)
print(f"分类准确率：\n{accuracy:.4f}")


# ====================== 用 2 个特征训练模型（专门用来画图） ======================
# 只取前两个特征！！！
X_2d = iris.data[:, :2]
x2_train, x2_test, y2_train, y2_test = train_test_split(X_2d, y, train_size=0.8, random_state=42)
print(X_2d.shape)
scaler_2d = StandardScaler()
x2_train_scaled = scaler_2d.fit_transform(x2_train)
print(x2_train_scaled.shape)
# 2D 模型（专门画决策边界）
clf_2d = SVC(kernel='rbf', C=1)
clf_2d.fit(x2_train_scaled, y2_train)
#可视化
#设置网格步长，用于生成绘制决策边界的网络
h=.02
#确定绘图区域的边界，分别根据训练数据中第一个特征（花萼长度）和第二个特征（花萼宽度）的最大值和最小值
x_min,x_max = x2_train_scaled[:,0].min()-1,x2_train_scaled[:,0].max()+1
y_min,y_max = x2_train_scaled[:,1].min()-1,x2_train_scaled[:,1].max()+1
xx,yy =np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
z = clf_2d.predict(np.c_[xx.ravel(),yy.ravel()])

#将预测结果z的形状重塑与表格xx相同的形状，以便后续绘制等高线图
z = z.reshape(xx.shape)
print(z.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm,alpha=0.8)
plt.scatter(x2_train_scaled[:,0],x2_train_scaled[:,1],c=y_train,cmap=plt.cm.coolwarm)
plt.title("SVM Classification on Iris Dataset")
plt.xlabel("Sepal length (Standardized)")
plt.ylabel("Sepal width (Standardized)")
plt.show()
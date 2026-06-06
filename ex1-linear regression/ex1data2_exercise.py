import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'data1/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
# 特征缩放
data2 = (data2 - data2.mean()) / data2.std()

def computeCost(X, y, theta):
    # 使用 @ 进行矩阵乘法：(47,3) @ (3,1) -> (47,1)
    inner = np.power((X @ theta.T - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    # 初始化为 array
    temp = np.zeros(theta.shape)
    parameters = theta.shape[1]
    cost = np.zeros(iters)

    for i in range(iters):
        # 使用 @ 替代 *，确保执行的是矩阵乘法
        error = (X @ theta.T) - y

        for j in range(parameters):
            # 提取一列时保持维度为 (47,1)，方便与 error 对应相乘
            term = np.multiply(error, X[:, j].reshape(-1, 1))
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

# 插入常数项列
data2.insert(0, 'Ones', 1)

# 分离特征和标签
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]

# --- 核心修改：统一使用 np.array ---
X2 = np.array(X2.values)
y2 = np.array(y2.values)
# 初始 theta 设为 (1, 3) 的二维 array，方便后续转置
theta2 = np.array([[0, 0, 0]])

alpha = 0.01
iters = 1000

# 运行梯度下降
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# 绘制 Cost 曲线
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

# 使用 sklearn 验证
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X2, y2)

# 绘图：X2[:, 1] 取出的是 Size 列
x = X2[:, 1]
f = model.predict(X2).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Size, data2.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Predicted Price vs. Size')
plt.show()
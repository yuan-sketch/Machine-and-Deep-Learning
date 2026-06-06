# 袁超
# 开发时间：2026/4/8 11:59
import numpy as np
import matplotlib.pyplot as plt
from utils2 import *
import math
import copy
#加载数据
X_train,y_train=load_data("data2/ex2data2.txt")
print("X_train的前五个数据：\n",X_train[:5])
print("y_train的前五个数据：\n",y_train[:5])

#可视化
#plot_data(X_train,y_train[:],pos_label="Admitted",neg_label="Not admitted")
#plt.ylabel("Score 2")
#plt.xlabel("Score 1")
#plt.legend(loc="upper right")
#plt.show()
#归一化函数
def feature(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm=(X-mu)/sigma
    return X_norm
#sigmoid函数
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))
#成本函数（正则化）
def compute_cost_reg(X,y,w,b,lambda_=1):
    m,n=X.shape
    cost=0.
    for i in range(m):
        z = np.dot(X[i], w) + b
        if z >= 0:
            f_wb_i = 1 / (1 + np.exp(-z))
        else:
            f_wb_i = np.exp(z) / (1 + np.exp(z))
        f_wb_i = max(min(f_wb_i, 0.99999999), 0.00000001)
        cost+=-y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)
    cost=cost/m
    reg_cost   =0.

    for j in range(n):
        reg_cost+=(w[j]**2)
    reg_cost=(lambda_/(2*m))*reg_cost
    total_cost=cost+reg_cost
    return total_cost
#梯度函数（正则化）
def compute_gradient_reg(X,y,w,b,lambda_=1):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0.
    for i in range(m):
        z = np.dot(X[i], w) + b
        if z >= 0:
            f_wb_i = 1 / (1 + np.exp(-z))
        else:
            f_wb_i = np.exp(z) / (1 + np.exp(z))
        dj_dw+=(f_wb_i-y[i])*X[i]
        dj_db+=(f_wb_i-y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    for j in range(n):
        dj_dw[j]=dj_dw[j]+(lambda_/m)*w[j]
    return dj_db,dj_dw
#梯度下降函数（正则化）
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            cost_val = np.atleast_1d(J_history[-1])[0]
            print(f"Iteration {i:4}: Cost {cost_val:8.4f}")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing
#预测函数
def predict(X, w, b, mu=None, sigma=None):
    # 如果传入了均值标准差，就先归一化
    if mu is not None and sigma is not None:
        X = (X - mu) / sigma
    return (sigmoid(np.dot(X, w) + b) >= 0.5).astype(float)

# Initialize fitting parameters
X_norm=map_feature(X_train[:,0],X_train[:,1])
#X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_norm.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ to 1 (you can try varying this)
lambda_ = 0.01;
# Some gradient descent settings
iterations = 100000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_norm, y_train, initial_w, initial_b,
                                    compute_cost_reg, compute_gradient_reg,
                                    alpha, iterations, lambda_)

plot_decision_boundary(w,b,X_norm,y_train)
#Compute accuracy on the training set
p = predict(X_norm, w, b)

print('训练准确率: %f'%(np.mean(p == y_train) * 100))


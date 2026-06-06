# 袁超
# 开发时间：2026/4/6 13:08

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

#load the dataset
x_train,y_train=load_data()
print("x_train的类型\n",type(x_train))
print("x_train的前五项是\n",x_train[:5])
print("y_train的类型\n",type(y_train))
print("y_train的前五项是\n",y_train[:5])

print("x_train的维度是\n",x_train.shape)
print("y_train的维度是\n",y_train.shape)
print("训练数据是\n",len(x_train))

#可视化数据
plt.scatter(x_train,y_train,marker='x',c='r')
#设置标题
plt.title("profit vs population")
#设置y轴坐标
plt.ylabel("profit in $10,000")
#设置x轴坐标
plt.xlabel("population of city in 10,000")

# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b):

    m = x.shape[0]

    # You need to return this variable correctly
    total_cost = 0
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_sum=cost_sum+cost
        total_cost=(1/(2*m))*cost_sum
    return total_cost

# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w (zeros): {cost:.3f}')

# Public tests

from public_tests import *
compute_cost_test(compute_cost)

def compute_gradient(x, y, w, b):
    dj_dw = 0
    dj_db = 0
    m = x.shape[0]
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=(f_wb-y[i])
        dj_dw+=dj_dw_i
        dj_db+=dj_db_i
    dj_dw=(1/m)*dj_dw
    dj_db=(1/m)*dj_db

    return dj_dw, dj_db

initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(x)

    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"迭代次数 {i:4}: 成本 {float(J_history[-1]):8.2f}   ")

    return w, b, J_history, w_history

initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b,
                     compute_cost, compute_gradient, alpha, iterations)
print("梯度下降得到的 w 和 b", w, b)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data1.
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()

predict1 = 3.5 * w + b
print('如果人口是 35,000, 预测利润是 $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('如果人口是 70,000, 预测利润是 $%.2f' % (predict2*10000))
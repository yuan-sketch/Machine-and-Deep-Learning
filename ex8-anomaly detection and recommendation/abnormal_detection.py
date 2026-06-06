# 袁超
# 开发时间：2026/4/16 14:17
import numpy as np
import matplotlib.pyplot as plt
from utils_ex8 import *
'''
#小样本示例
X_train,X_val,y_val = load_data('data/ex8data1.mat')
print("X_train的前五项元素：\n",X_train[:5])
print("X_val的前五项元素：\n",X_val[:5])
print("y_train的前五项元素：\n",y_val[:5])

print ('The shape of X_train is:', X_train.shape)
print ('The shape of X_val is:', X_val.shape)
print ('The shape of y_val is: ', y_val.shape)

plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b')

# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()

mu,var = estimate_guassian(X_train)
print("X_train的平均值\n",mu)
print("X_train的方差\n",var)

p = multivariate_gaussian(X_train, mu, var)

#Plotting code
#visualize_fit(X_train, mu, var)
p_val = multivariate_gaussian(X_val,mu,var)
epsilon,F1 = select_threshold(y_val,p_val)

print("best epsilon:%e"%epsilon)
print("best F1:%f"%F1)

# Find the outliers in the training set
outliers = p < epsilon

# Visualize the fit
visualize_fit(X_train, mu, var)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 13,markerfacecolor='none', markeredgewidth=2)
plt.show()
'''
#大样本示例
def load_data_multi(filename):
    data = loadmat(filename)
    X_train_high = data['X']
    X_val_high = data['Xval']
    y_val_high = data['yval'].flatten()
    return X_train_high,X_val_high,y_val_high

X_train_high,X_val_high,y_val_high = load_data_multi('data/ex8data2.mat')

print("X_train_high的shape是：",X_train_high.shape)
print("X_val_high的shape是：",X_val_high.shape)
print("y_val_high的shape是：",y_val_high.shape)

mu_high,var_high = estimate_guassian(X_train_high)
p_high = multivariate_gaussian(X_train_high,mu_high,var_high)
p_val_high = multivariate_gaussian(X_val_high,mu_high,var_high)
epsilon_high,F1_high = select_threshold(y_val_high,p_val_high)
print("best epsilon:%e"%epsilon_high)
print("best F1:%f"%F1_high)
print("异常值总数：%d"%sum(p_high<epsilon_high))
# 袁超
# 开发时间：2026/4/13 18:25
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning1.mplstyle')
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.keras.backend.set_floatx('float64')
from assigment_utils import *

tf.autograph.set_verbosity(0)

#划分数据集，利用train_test_split函数
X,y,x_ideal,y_ideal = gen_data(18,2,0.7)
print("X.shape",X.shape,"y.shape",y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
print("X_train.shape",X_train.shape,"y_train.shape",y_train.shape)
print("X_test.shape",X_test.shape,"y_test.shape",y_test.shape)

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()

def plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal,
                       err_train, err_cv, optimal_degree, max_degree):
    """
    绘制：误差曲线 + 最优阶数拟合曲线
    """
    plt.figure(figsize=(10, 4))

    # 子图1：训练误差 & 交叉验证误差曲线
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, max_degree+1), err_train, marker='o', label='train err')
    plt.plot(np.arange(1, max_degree+1), err_cv, marker='o', label='cv err')
    plt.axvline(x=optimal_degree, color='red', linestyle='--', label=f'optimal={optimal_degree}')
    plt.xlabel('degree')
    plt.ylabel('MSE')
    plt.title('Train & CV Error')
    plt.legend()
    plt.grid(True)

    # 子图2：最优模型拟合曲线
    plt.subplot(1, 2, 2)
    plt.plot(x_ideal, y_ideal, '--', c='orangered', label='ideal')
    plt.plot(x, y_pred[:, optimal_degree-1], c='red', label='optimal fit')
    plt.scatter(X_train, y_train, c='red', s=15, label='train')
    plt.scatter(X_cv, y_cv, c='#ff9300', s=15, label='cv')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Optimal degree = {optimal_degree}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range):
    """
    绘制：正则化调优曲线 + 最优正则化拟合曲线
    """
    plt.figure(figsize=(10, 4))

    # 子图1：训练误差 & 交叉验证误差随正则化强度变化
    plt.subplot(1, 2, 1)
    plt.plot(lambda_range, err_train, marker='o', label='train err')
    plt.plot(lambda_range, err_cv, marker='o', label='cv err')
    plt.axvline(x=lambda_range[optimal_reg_idx], color='red', linestyle='--',
                label=f'optimal λ = {lambda_range[optimal_reg_idx]}')
    plt.xscale('log')  # 正则化通常用对数坐标更清晰
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('MSE')
    plt.title('Train & CV Error vs Regularization')
    plt.legend()
    plt.grid(True)

    # 子图2：最优正则化模型拟合曲线
    plt.subplot(1, 2, 2)
    plt.plot(x, y_pred[:, optimal_reg_idx], c='red', label='optimal fit')
    plt.scatter(X_train, y_train, c='red', s=15, label='train')
    plt.scatter(X_cv, y_cv, c='#ff9300', s=15, label='cv')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Optimal Regularization λ = {lambda_range[optimal_reg_idx]:.1e}')
    plt.legend()
    plt.tight_layout()
    plt.show()
# 修复升级后的 lin_model 类（支持正则化）

def tune_m():
    """
    调参：不同训练集大小 m 对误差的影响（高方差/高偏差诊断）
    返回：X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree
    """
    # 生成数据
    X, y, x_ideal, y_ideal = gen_data(200, 5, 0.7)

    # 拆分：训练集 + 交叉验证集
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=1)

    degree = 10  # 使用高次多项式制造过拟合，观察学习曲线
    m_samples = [1, 2, 4, 8, 16, 24, 32, 40, 48]  # 训练样本数量
    m_range = np.array(m_samples)
    n = len(m_samples)

    err_train = np.zeros(n)
    err_cv = np.zeros(n)

    # 绘图用 x
    x = np.linspace(0, int(X.max()), 100).reshape(-1, 1)
    y_pred = np.zeros((100, n))

    # 遍历不同训练集大小
    for i, m in enumerate(m_samples):
        # 取前 m 个样本作为训练集
        X_sub = X_train[:m]
        y_sub = y_train[:m]

        # 高次多项式 + 正则化（固定小正则化，制造过拟合）
        lmodel = lin_model(degree, regularization=True, lambda_=0.001)
        lmodel.fit(X_sub, y_sub)

        # 计算误差
        yhat_train = lmodel.predict(X_sub)
        err_train[i] = lmodel.mse(y_sub, yhat_train)

        yhat_cv = lmodel.predict(X_cv)
        err_cv[i] = lmodel.mse(y_cv, yhat_cv)

        # 保存预测曲线
        y_pred[:, i] = lmodel.predict(x).ravel()

    return X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree


def plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree):
    """
    绘制学习曲线：训练集大小 m 对 train/cv 误差的影响
    """
    plt.figure(figsize=(10, 4))

    # 子图1：学习曲线（核心！判断高偏差 / 高方差）
    plt.subplot(1, 2, 1)
    plt.plot(m_range, err_train, marker='o', label='Train error')
    plt.plot(m_range, err_cv, marker='o', label='CV error')
    plt.xlabel('Number of training examples (m)')
    plt.ylabel('MSE')
    plt.title(f'Learning Curve (degree={degree})')
    plt.grid(True)
    plt.legend()

    # 子图2：最大训练集下的拟合效果
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, c='red', s=15, label='Train')
    plt.scatter(X_cv, y_cv, c='#ff9300', s=15, label='CV')
    plt.plot(x, y_pred[:, -1], c='red', lw=1.5, label='Model fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fit with max training size')
    plt.legend()

    plt.tight_layout()
    plt.show()
class lin_model:
    def __init__(self, degree, regularization=False, lambda_=0.0):
        self.degree = degree
        self.regularization = regularization
        self.lambda_ = lambda_

        # 多项式特征 + 标准化
        self.poly = PolynomialFeatures(degree, include_bias=False)
        self.scaler = StandardScaler()

        # 选择模型：普通线性回归 或 带正则化的岭回归
        if regularization:
            self.model = Ridge(alpha=lambda_)  # L2正则化
        else:
            self.model = LinearRegression()

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

    def mse(self, y, yhat):
        return np.mean((y - yhat) ** 2)
def eval_mse(y,yhat):
    m = len(y)
    err = 0.0
    for i in range(m):
        err_i = ((y-yhat)**2)
        err+=err_i
    err = err/(2*m)
    return(err)

def plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree):
    plt.figure(figsize=(6,4))
    plt.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    plt.plot(x, y_pred, color="red", label="prediction", lw=1)
    plt.scatter(X_train, y_train, color="red", label="train")
    plt.scatter(X_test, y_test, color=dlc["dlblue"], label="test")
    plt.title(f"Polynomial Fit (degree={degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


degree = 10
lmodel = lin_model(degree)
lmodel.fit(X_train,y_train)
yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train,yhat)
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test,yhat)
print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

# plot predictions over data range
x = np.linspace(0,int(X.max()),100).reshape(-1,1)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)

plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)

X,y,x_ideal,y_ideal = gen_data(40,5,0.7)
print("X.shape",X.shape,"y.shape",y.shape)

X_train,X_,y_train,y_ = train_test_split(X,y,test_size=0.40,random_state=1)
X_cv,X_test,y_cv,y_test = train_test_split(X_,y_,test_size=0.5,random_state=1)
'''print("X_train.shape",X_train.shape,"y_train.shape",y_train.shape)
print("X_cv.shape",X_cv.shape,"y_cv.shape",y_cv.shape)
print("X_test.shape",X_test.shape,"y_test.shape",y_test.shape)

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, CV, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_cv, y_cv,       color = dlc["dlorange"], label="cv")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()'''


max_degree = 9
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)
x = np.linspace(0, int(X.max()), 100).reshape(-1,1)
y_pred = np.zeros((100, max_degree))  # columns are lines to plot

for degree in range(max_degree):
    lmodel = lin_model(degree + 1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)
    y_pred[:, degree] = lmodel.predict(x)

optimal_degree = np.argmin(err_cv) + 1

plt.close("all")
plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal,
                   err_train, err_cv, optimal_degree, max_degree)

#调整正则化
lambda_range = np.array([0.0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100])
num_steps = len(lambda_range)
degree = 10
err_train = np.zeros(num_steps)
err_cv = np.zeros(num_steps)
x = np.linspace(0,int(X.max()),100).reshape(-1,1)
y_pred = np.zeros((100,num_steps))

for i in range(num_steps):
    lambda_ = lambda_range[i]
    lmodel = lin_model(degree,regularization=True,lambda_=lambda_)
    lmodel.fit(X_train,y_train)
    yhat = lmodel.predict(X_train)
    err_train[i]=lmodel.mse(y_train,yhat)
    yhat=lmodel.predict(X_cv)
    err_cv[i]=lmodel.mse(y_cv,yhat)
    y_pred[:,i]=lmodel.predict(x)

optimal_reg_idx = np.argmin(err_cv)

plt.close("all")
plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)

X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)
# 袁超
# 开发时间：2026/4/16 14:34
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ====================== 解决 matplotlib 中文显示（宋体）======================
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


# ====================== 1. 加载 .mat 数据 ======================
def load_data(filename):
    """
    加载吴恩达作业 .mat 格式数据
    返回 X, X_val, y_val（训练集、验证集、标签）
    """
    data = loadmat(filename)
    X = data['X']
    X_val = data['Xval']
    y_val = data['yval'].flatten()  # 转为一维
    return X, X_val, y_val


# ====================== 2. 绘制数据集散点图 ======================
def plot_dataset(X, title='数据集分布', xlabel='特征1', ylabel='特征2'):
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='blue', s=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


# ====================== 3. 绘制高斯分布等高线图 ======================
def plot_contours(X, mu, sigma2, title='高斯分布等高线'):
    """
    根据 mu, sigma2 绘制高斯分布等高线
    """
    x = np.linspace(0, 30, 100)
    y = np.linspace(0, 30, 100)
    X_mesh, Y_mesh = np.meshgrid(x, y)

    # 展平成二维点
    points = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

    # 计算每个点的高斯概率
    p = np.exp(-np.sum((points - mu) ** 2 / (2 * sigma2), axis=1))
    p = p.reshape(X_mesh.shape)

    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='blue', s=30)

    # 画等高线
    cont_levels = [10 ** i for i in range(-20, 0, 3)]
    plt.contour(X_mesh, Y_mesh, p, levels=cont_levels)

    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.grid(True)
    plt.show()


# ====================== 4. 绘制异常检测结果（正常/异常点区分）======================
def plot_anomalies(X, y_val, y_pred, title='异常检测结果'):
    """
    y_pred=1 表示异常，标为红色
    y_pred=0 表示正常，标为蓝色
    """
    plt.figure(figsize=(6, 4))

    # 正常点
    normal = X[y_pred == 0]
    plt.scatter(normal[:, 0], normal[:, 1], marker='x', c='blue', label='正常', s=30)

    # 异常点
    anomalies = X[y_pred == 1]
    plt.scatter(anomalies[:, 0], anomalies[:, 1], marker='o', c='red', s=80,
                facecolors='none', edgecolors='r', label='异常')

    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True)
    plt.show()


# ====================== 5. 计算高斯分布概率 ======================
def multivariate_gaussian(X, mu, sigma2):
    """
    计算多元高斯分布概率（对角协方差，即各特征独立）
    """
    m, n = X.shape
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)

    X_norm = X - mu
    p = (1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma2) ** 0.5)) \
        * np.exp(-0.5 * np.sum(np.dot(X_norm, np.linalg.inv(sigma2)) * X_norm, axis=1))
    return p

import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示（宋体）
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

def visualize_fit(X, mu, var):
    """
    可视化多元高斯分布的等高线（轮廓图）
    输入：
        X  - 数据 (m, 2)
        mu - 均值 (2,)
        var - 方差 (2,)
    """
    # 生成网格坐标
    x = np.linspace(0, 30, 100)
    y = np.linspace(0, 30, 100)
    X_mesh, Y_mesh = np.meshgrid(x, y)

    # 展平成二维点
    points = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

    # 计算高斯概率
    m, n = points.shape
    p = np.zeros(m)
    for i in range(m):
        p[i] = np.prod((1 / np.sqrt(2 * np.pi * var)) * np.exp(-(points[i] - mu)**2 / (2 * var)))

    # 恢复成网格形状
    p = p.reshape(X_mesh.shape)

    # 画图
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='blue', s=40)

    # 画等高线（对数间隔，让轮廓更清晰）
    levels = [10**i for i in range(-20, 0, 3)]
    plt.contour(X_mesh, Y_mesh, p, levels=levels, colors='r')

    plt.title('高斯分布拟合结果', fontsize=14)
    plt.xlabel('Latency(ms)', fontsize=12)
    plt.ylabel('Throughput(mb/s)', fontsize=12)
    plt.grid(True)
    #plt.show()

#估计高斯的参数
def estimate_guassian(x):
    m,n = x.shape
    mu = np.sum(x,axis=0)/m
    var = (np.sum((x-mu)**2,axis=0))/m#axis=0是按照列相加，axis=1是按照行相加
    return mu,var

#选择阈值
def select_threshold(y_val,p_val):
    best_spsilon = 0
    best_F1 = 0
    F1 = 0
    step_size = (max(p_val)-min(p_val))/1000
    for epsilon in np.arange(min(p_val),max(p_val),step_size):
        predictions = (p_val<epsilon)
        fp = sum((predictions==1)&(y_val==0))
        tp = sum((predictions==1)&(y_val==1))
        fn = sum((predictions==0)&(y_val==1))
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)

        if F1>best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return (best_epsilon,best_F1)
# 袁超
# 开发时间：2026/4/9 15:25
import numpy as np
from scipy.io import loadmat


def load_data():
    # 正确读取 .mat 文件
    data = loadmat("data3/ex3data1.mat")

    # 取出 X 和 y
    X = data['X']  # 特征 (5000, 400)
    y = data['y']  # 标签 (5000, 1)

    # 取前1000条数据
    X = X[:1000]
    y = y[:1000]

    return X, y

def load_data_all():
    # 正确读取 .mat 文件
    data = loadmat("data3/ex3data1.mat")

    # 取出 X 和 y
    X = data['X']  # 特征 (5000, 400)
    y = data['y']  # 标签 (5000, 1)
    return X, y
def load_weights():
    # 读取 .mat 权重文件
    weights = loadmat("data3/ex3weights.mat")

    # Theta1, Theta2
    w1 = weights['Theta1']  # (25, 401)
    w2 = weights['Theta2']  # (10, 26)

    # 手动拆分 w1, w2 里的 权重 和 偏置
    b1 = w1[:, 0]  # 偏置
    w1 = w1[:, 1:]  # 权重

    b2 = w2[:, 0]  # 偏置
    w2 = w2[:, 1:]  # 权重

    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
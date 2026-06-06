# -*- coding: utf-8 -*-
"""
吴恩达《Machine Learning》课程数据集（sklearn.datasets 风格封装）
项目来源：和鲸社区
转载使用请务必标注来源：和鲸社区
"""
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from pathlib import Path

__all__ = [
    "load_ng_ex1",  # 单变量线性回归
    "load_ng_ex1_multi",  # 多变量线性回归
    "load_ng_ex2",  # 逻辑回归二分类（录取）
    "load_ng_ex2_reg",  # 逻辑回归二分类（芯片，带正则）
    "load_ng_ex3",  # 多分类（手写数字 0-9）
    "load_ng_ex4",  # 神经网络（手写数字）
    "load_ng_ex5",  # 偏差/方差（水库水位）
    "load_ng_ex6",  # SVM（线性/高斯/垃圾邮件）
    "load_ng_ex7",  # K-Means / PCA（图像压缩）
    "load_ng_ex8",  # 异常检测 / 推荐系统
]


def _load_txt_data(path, delimiter=",", dtype=float):
    """内部工具：加载txt/csv数据"""
    data = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
    return data


def _make_bunch(X, y, feature_names=None, target_names=None, DESCR=""):
    """内部工具：生成sklearn风格Bunch对象"""
    return Bunch(
        data=X,
        target=y,
        feature_names=feature_names or [],
        target_names=target_names or [],
        DESCR=DESCR
    )


# ------------------------------
# 练习1：线性回归 Linear Regression
# ------------------------------
def load_ng_ex1():
    """
    吴恩达ML ex1：单变量线性回归（城市人口 vs 餐饮利润）
    数据：ex1data1.txt
    特征：城市人口 (10k)
    标签：利润 (10k$)
    """
    data = _load_txt_data("ex1data1.txt")
    X = data[:, 0:1]
    y = data[:, 1]
    desc = """吴恩达ML课程 ex1 单变量线性回归数据集
    97个样本，1个特征：城市人口（单位：万人）
    目标：预测餐饮利润（单位：万美元）"""
    return _make_bunch(X, y,
                       feature_names=["population"],
                       target_names=["profit"],
                       DESCR=desc)


def load_ng_ex1_multi():
    """
    吴恩达ML ex1：多变量线性回归（房屋面积、卧室数 vs 房价）
    数据：ex1data2.txt
    """
    data = _load_txt_data("ex1data2.txt")
    X = data[:, 0:2]
    y = data[:, 2]
    desc = """吴恩达ML课程 ex1 多变量线性回归数据集
    47个样本，2个特征：房屋面积(平方英尺)、卧室数量
    目标：预测房屋价格"""
    return _make_bunch(X, y,
                       feature_names=["square_feet", "bedrooms"],
                       target_names=["price"],
                       DESCR=desc)


# ------------------------------
# 练习2：逻辑回归 Logistic Regression
# ------------------------------
def load_ng_ex2():
    """
    吴恩达ML ex2：逻辑回归（两次考试成绩 → 是否录取）
    数据：ex2data1.txt
    """
    data = _load_txt_data("ex2data1.txt")
    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    desc = """吴恩达ML课程 ex2 逻辑回归二分类（录取）
    100个样本，2个特征：考试1、考试2分数
    标签：1=录取，0=未录取"""
    return _make_bunch(X, y,
                       feature_names=["exam1", "exam2"],
                       target_names=["not_admitted", "admitted"],
                       DESCR=desc)


def load_ng_ex2_reg():
    """
    吴恩达ML ex2：正则化逻辑回归（芯片测试 → 是否合格）
    数据：ex2data2.txt
    """
    data = _load_txt_data("ex2data2.txt")
    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    desc = """吴恩达ML课程 ex2 正则化逻辑回归（芯片质量）
    118个样本，2个特征：芯片测试1、测试2结果
    标签：1=合格，0=不合格"""
    return _make_bunch(X, y,
                       feature_names=["test1", "test2"],
                       target_names=["rejected", "accepted"],
                       DESCR=desc)


# ------------------------------
# 练习3：多分类与神经网络（手写数字）
# ------------------------------
def load_ng_ex3():
    """
    吴恩达ML ex3：多分类（手写数字 0-9）
    数据：ex3data1.mat（5000张 20x20=400像素）
    注意：需先下载 ex3data1.mat 并放在同目录
    """
    from scipy.io import loadmat
    mat = loadmat("ex3data1.mat")
    X = mat["X"]  # (5000, 400)
    y = mat["y"].flatten()
    y[y == 10] = 0  # 原数据10表示0，修正为0
    desc = """吴恩达ML课程 ex3 多分类（手写数字0-9）
    5000个样本，400个特征（20x20像素展平）
    标签：0-9 数字"""
    return _make_bunch(X, y,
                       feature_names=[f"pixel_{i}" for i in range(400)],
                       target_names=[str(i) for i in range(10)],
                       DESCR=desc)


# ------------------------------
# 练习4：神经网络（反向传播）
# ------------------------------
def load_ng_ex4():
    """同ex3数据（手写数字），用于神经网络训练"""
    return load_ng_ex3()


# ------------------------------
# 练习5：偏差与方差（正则化线性回归）
# ------------------------------
def load_ng_ex5():
    """
    吴恩达ML ex5：偏差/方差（水库水位变化 → 水流量）
    数据：ex5data1.mat
    """
    from scipy.io import loadmat
    mat = loadmat("ex5data1.mat")
    X = mat["X"]
    y = mat["y"].flatten()
    Xval = mat["Xval"]
    yval = mat["yval"].flatten()
    Xtest = mat["Xtest"]
    ytest = mat["ytest"].flatten()
    bunch = _make_bunch(X, y,
                        feature_names=["water_level"],
                        target_names=["flow"],
                        DESCR="吴恩达ML ex5 偏差方差数据集")
    # 额外添加验证/测试集
    bunch.Xval = Xval
    bunch.yval = yval
    bunch.Xtest = Xtest
    bunch.ytest = ytest
    return bunch


# ------------------------------
# 练习6：支持向量机 SVM
# ------------------------------
def load_ng_ex6(part=1):
    """
    吴恩达ML ex6：SVM数据集
    part=1: 线性可分 (ex6data1.txt)
    part=2: 非线性 (ex6data2.txt)
    part=3: 垃圾邮件特征 (ex6data3.txt)
    """
    if part == 1:
        data = _load_txt_data("ex6data1.txt")
    elif part == 2:
        data = _load_txt_data("ex6data2.txt")
    elif part == 3:
        data = _load_txt_data("ex6data3.txt")
    else:
        raise ValueError("part must be 1/2/3")

    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    desc = f"吴恩达ML ex6 SVM 数据集 part{part}"
    return _make_bunch(X, y,
                       feature_names=["x1", "x2"],
                       target_names=["class0", "class1"],
                       DESCR=desc)


# ------------------------------
# 练习7：K-Means & PCA
# ------------------------------
def load_ng_ex7(kind="kmeans"):
    """
    吴恩达ML ex7：聚类/PCA
    kind='kmeans': ex7data1.txt / ex7data2.txt
    kind='pca': ex7data1.txt
    kind='image': 图像压缩用（需bird_small.mat）
    """
    if kind == "kmeans":
        data = _load_txt_data("ex7data2.txt")
        X = data
        y = None
    elif kind == "pca":
        data = _load_txt_data("ex7data1.txt")
        X = data
        y = None
    elif kind == "image":
        from scipy.io import loadmat
        mat = loadmat("bird_small.mat")
        X = mat["A"]
        y = None
    else:
        raise ValueError("kind must be kmeans/pca/image")

    return _make_bunch(X, y,
                       feature_names=[f"f{i}" for i in range(X.shape[1])],
                       DESCR=f"吴恩达ML ex7 {kind} 数据集")


# ------------------------------
# 练习8：异常检测 & 推荐系统
# ------------------------------
def load_ng_ex8(kind="anomaly"):
    """
    吴恩达ML ex8：异常检测 / 推荐系统
    kind='anomaly': 服务器特征 (ex8data1.mat)
    kind='recommend': 电影评分 (ex8_movies.mat)
    """
    from scipy.io import loadmat
    if kind == "anomaly":
        mat = loadmat("ex8data1.mat")
        X = mat["X"]
        y = mat["yval"].flatten()
        return _make_bunch(X, y,
                           feature_names=["latency", "throughput"],
                           target_names=["normal", "anomaly"],
                           DESCR="吴恩达ML ex8 异常检测数据集")
    elif kind == "recommend":
        mat = loadmat("ex8_movies.mat")
        Y = mat["Y"]
        R = mat["R"]
        bunch = Bunch(data=Y, mask=R,
                      DESCR="吴恩达ML ex8 推荐系统电影评分数据集")
        return bunch
    else:
        raise ValueError("kind must be anomaly/recommend")
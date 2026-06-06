# 袁超
# 开发时间：2026/4/13 18:27
"""
assignment_utils.py
contains routines used by C2_W3 Assignments
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.widgets import Button, CheckButtons
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

from ipywidgets import Output
np.set_printoptions(precision=2)

# ======================
# 颜色定义（你提供的部分）
# ======================
dlc = dict(
    dlblue='#0096ff',
    dlorange='#FF9300',
    dldarkred='#C00000',
    dlmagenta='#FF40FF',
    dlpurple='#7030A0',
    dldarkblue='#0D5BDC'
)
dlblue = '#0096ff'
dlorange = '#FF9300'
dldarkred = '#C00000'
dlmagenta = '#FF40FF'
dlpurple = '#7030A0'
dldarkblue = '#0D5BDC'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

# ======================
# 绘图风格设置
# ======================
plt.style.use('./deeplearning1.mplstyle')

# ======================
# 1. 回归相关工具函数
# ======================
def gen_data(m, seed=1, scale=0.7):
    np.random.seed(seed)
    x = np.linspace(0, 1, m).reshape(-1, 1)
    y = 2 * np.cos(x * 6) + 3 * x + np.random.randn(m, 1) * scale
    x_ideal = np.linspace(0, 1, 100).reshape(-1, 1)
    y_ideal = 2 * np.cos(x_ideal * 6) + 3 * x_ideal
    return x, y.flatten(), x_ideal, y_ideal.flatten()

def gen_multi_data():
    x = np.linspace(0, 20, 100)
    y1 = 0.5 * x + 1 + 2 * np.sin(x) + 0.5 * (np.random.rand(100) * 3 - 1.5)
    y2 = 2 * x + 1 + 2 * np.sin(x) + 0.5 * (np.random.rand(100) * 3 - 1.5)
    y3 = 3 * x + 1 + 2 * np.sin(x) + 0.5 * (np.random.rand(100) * 3 - 1.5)
    return x, y1, y2, y3

def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title=""):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color=dlblue, label="Train")
    plt.scatter(x_cv, y_cv, color=dlorange, label="CV")
    plt.scatter(x_test, y_test, color=dlmagenta, label="Test")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_cv_mses(degrees, train_mses, cv_mses, title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_mses, marker='o', c=dlblue, label="Train MSE")
    plt.plot(degrees, cv_mses, marker='o', c=dlorange, label="CV MSE")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ======================
# 2. 分类/决策边界绘图
# ======================
def plot_cat_decision_boundary(X, y, predict, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(dlcolors))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(dlcolors))
    plt.title(title)
    plt.show()

# ======================
# 3. 神经网络/正则化绘图
# ======================
def plot_regression_models(x, y, models, scalers, degrees, title=""):
    x_cont = np.linspace(0, 20, 100)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='gray', s=20)
    for i, (d, m, s) in enumerate(zip(degrees, models, scalers)):
        X_cont_poly = PolynomialFeatures(degree=d, include_bias=False).fit_transform(x_cont.reshape(-1, 1))
        X_cont_poly_scaled = s.transform(X_cont_poly)
        y_pred = m.predict(X_cont_poly_scaled)
        plt.plot(x_cont, y_pred, label=f"Degree {d}", color=dlcolors[i % len(dlcolors)])
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ======================
# 4. 学习曲线绘制
# ======================
def plot_learning_curve(train_mses, cv_mses, title="Learning Curve"):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(train_mses)+1), train_mses, c=dlblue, label="Train")
    plt.plot(np.arange(1, len(cv_mses)+1), cv_mses, c=dlorange, label="CV")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ======================
# 5. 交互/按钮工具
# ======================
def create_button(ax, label, callback):
    button = Button(ax, label, color='lightgoldenrodyellow', hovercolor='0.975')
    button.on_clicked(callback)
    return button

# ======================
# 6. 辅助工具
# ======================
def print_mse(train_mse, cv_mse):
    print(f"Train MSE: {train_mse:.2f}")
    print(f"CV MSE: {cv_mse:.2f}")

def base_fig():
    fig, ax = plt.subplots(figsize=(8, 5))
    return fig, ax
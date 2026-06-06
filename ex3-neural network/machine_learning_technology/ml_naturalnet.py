# 袁超
# 开发时间：2026/4/14 12:18
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.keras.backend.set_floatx('float64')
from assigment_utils import *

tf.autograph.set_verbosity(0)
from matplotlib.patches import Circle


def gen_blobs(random_state=1):
    np.random.seed(random_state)
    classes = 6
    total_samples = 800
    std = 0.4

    centers = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [-2.0, 1.0],
        [-2.0, -1.0]
    ])

    counts = [133, 133, 133, 134, 134, 133]
    X, y = [], []

    for i, cnt in enumerate(counts):
        x = centers[i, 0] + np.random.randn(cnt) * std
        y_col = centers[i, 1] + np.random.randn(cnt) * std
        X.append(np.column_stack([x, y_col]))
        y.append(np.full(cnt, i))

    X = np.concatenate(X)
    y = np.concatenate(y).astype(int)
    return X, y, centers, classes, std


def plt_train_eq_dist(X_train, y_train, classes, X_cv, y_cv, centers, std):
    colors = [
        '#1f77b4',
        '#2ca02c',
        '#ff7f0e',
        '#9467bd',
        '#d62728',
        '#8c564b'
    ]
    markers_train = ['o'] * 6
    markers_cv = ['<'] * 6
    labels = [f'c{i}' for i in range(classes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=120)

    # 左图
    ax1.set_title('Training, CV Data', fontsize=18)
    ax1.set_xlim(-3.2, 2.2)
    ax1.set_ylim(-3.2, 2.2)
    ax1.grid(False)

    for i in range(classes):
        idx_train = y_train == i
        ax1.scatter(X_train[idx_train, 0], X_train[idx_train, 1],
                    color=colors[i], marker='o', s=100, alpha=0.8, label=f'{labels[i]} train')

        idx_cv = y_cv == i
        ax1.scatter(X_cv[idx_cv, 0], X_cv[idx_cv, 1],
                    color=colors[i], marker='<', s=80, alpha=0.5, label=f'{labels[i]} cv')

    for i, c in enumerate(centers):
        ax1.add_patch(Circle(c, std * 3, color=colors[i], fill=False, lw=2))

    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_aspect('equal')

    # 右图
    ax2.set_title('ideal performance', fontsize=18)
    ax2.set_xlim(-3.2, 2.2)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(False)

    X_all = np.vstack([X_train, X_cv])
    y_all = np.hstack([y_train, y_cv])

    for i in range(classes):
        idx = y_all == i
        ax2.scatter(X_all[idx, 0], X_all[idx, 1],
                    color=colors[i], marker='o', s=100, alpha=0.9,label=labels[i])

    # 粉色理想边界
    #ax2.axvline(-1, color='magenta', lw=2)
    #ax2.axhline(0, color='magenta', lw=2)
    ax2.plot([-1, -1], [1, 2.5], color='magenta', lw=2)
    ax2.plot([-1, -1], [-1, -2.5], color='magenta', lw=2)
    ax2.plot([-3, -2], [0, 0], color='magenta', lw=2)
    ax2.plot([-1, 2], [-1, 2], color='magenta', lw=2)
    ax2.plot([-2, -1], [0, -1], color='magenta', lw=2)
    ax2.plot([-2, -1], [0, 1], color='magenta', lw=2)
    ax2.plot([-1, 2], [1, -2], color='magenta', lw=2)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plt_nn(model_predict, X_train, y_train, classes, X_cv, y_cv, suptitle="Complex Model"):
    """
    绘制神经网络分类结果：训练集 & 验证集 + 决策边界
    参数：model_predict -> 模型预测函数
    """
    # 配色和你之前完全一致
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728', '#8c564b']

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    fig.suptitle(suptitle, fontsize=18)

    # 坐标轴范围（和你数据匹配）
    x_min, x_max = -3.2, 2.2
    y_min, y_max = -2.5, 2.5

    # 生成网格，画决策背景
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = model_predict(grid)
    pred = pred.reshape(xx.shape)

    # ---------------------- 左图：训练集 ----------------------
    ax1.set_title("Training set")
    ax1.contourf(xx, yy, pred, alpha=0.2, levels=classes - 1, cmap='viridis')
    for i in range(classes):
        idx = y_train == i
        ax1.scatter(X_train[idx, 0], X_train[idx, 1],
                    c=colors[i], edgecolor='black', s=80, label=f'c{i}')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')

    # ---------------------- 右图：验证集 ----------------------
    ax2.set_title("CV set")
    ax2.contourf(xx, yy, pred, alpha=0.2, levels=classes - 1, cmap='viridis')
    for i in range(classes):
        idx = y_cv == i
        ax2.scatter(X_cv[idx, 0], X_cv[idx, 1],
                    c=colors[i], edgecolor='black', s=80, label=f'c{i}')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

# ===================== 主程序 =====================
X, y, centers, classes, std = gen_blobs()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=1, stratify=y
)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=1, stratify=y_temp
)

print("X_train.shape", X_train.shape, "X_cv.shape", X_cv.shape, "X_test.shape", X_test.shape)

#plt_train_eq_dist(X_train, y_train, classes, X_cv, y_cv, centers, std)


def eval_cat_err(y,yhat):
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i]!=y[i]:
            incorrect+=1
    cerr = incorrect/m
    return(cerr)

'''y_hat = np.array([1, 2, 0])
y_tmp = np.array([1, 2, 3])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333" )
y_hat = np.array([[1], [2], [0], [3]])
y_tmp = np.array([[1], [2], [1], [3]])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250" )
'''
#complex model
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ###
        Dense(120,activation='relu',name='layer1'),
        Dense(40,activation='relu',name='layer2'),
        Dense(6,activation='linear',name='layer3')
        ### END CODE HERE ###

    ], name="Complex"
)
model.compile(
    ### START CODE HERE ###
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
    ### END CODE HERE ###
)

model.fit(
    X_train,y_train,
    epochs=1000
)

model.summary()

#make a model for plotting routines to call
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")

training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")

#simple model

tf.random.set_seed(1234)
model_s = Sequential(
    [
        ### START CODE HERE ###
        Dense(6,activation='relu',name='l1'),
        Dense(6,activation='linear',name='l2')

        ### END CODE HERE ###
    ], name = "Simple"
)
model_s.compile(
    ### START CODE HERE ###
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
    ### START CODE HERE ###
)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# BEGIN UNIT TEST
model_s.fit(
    X_train,y_train,
    epochs=1000
)
model_s.summary()

#make a model for plotting routines to call
model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict_s,X_train,y_train, classes, X_cv, y_cv, suptitle="Simple Model")

#计算分类误差
training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
print(f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )


#正则化
tf.random.set_seed(1234)
model_r = Sequential(
    [
        ### START CODE HERE ###
        Dense(120,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.1),name='l1'),
        Dense(40,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.1),name='l2'),
        Dense(6,activation='linear',name='l3'),
        ### START CODE HERE ###
    ], name= None
)
model_r.compile(
    ### START CODE HERE ###
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
    ### START CODE HERE ###
)
model_r.fit(
    X_train,y_train,
    epochs=1000
)
model_r.summary()

# make a model for plotting routines to call
model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(), axis=1)

plt_nn(model_predict_r, X_train, y_train, classes, X_cv, y_cv, suptitle="Regularized")


training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )


tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)
for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")
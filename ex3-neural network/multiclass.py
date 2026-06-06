# 袁超
# 开发时间：2026/4/13 14:47
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib.colors import ListedColormap
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def plt_mc(X, y, classes, centers, std):
    # 颜色和标记
    cmp = ListedColormap(['blue', 'orange', 'green', 'purple'])
    plt.figure(figsize=(6, 4))

    # 绘制每个类别的点
    for i in range(classes):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    color=cmp(i), label=f"class {i}", s=30)

    plt.title("Multiclass Data")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
# 绘制多分类决策边界函数（吴恩达风格 plt_cat_mc）
def plt_cat_mc(X, y, model, classes):
    plt.figure(figsize=(6, 4))
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.02),
                         np.arange(ymin, ymax, 0.02))

    # 绘制神经网络单层 ReLU 层输出特征空间（吴恩达课程 plt_layer_relu）
    def plt_layer_relu(X, y, layer1, classes):
        # 获取第一层ReLU激活后的输出
        a1 = layer1(X).numpy()  # (样本数, 2)

        cmp = ListedColormap(['blue', 'orange', 'green', 'purple'])
        plt.figure(figsize=(6, 4))

        for i in range(classes):
            plt.scatter(a1[y == i, 0], a1[y == i, 1],
                        color=cmp(i), label=f'class {i}', s=40)

        plt.title("Layer 1 Output (ReLU Activated)")
        plt.xlabel("a1_0")
        plt.ylabel("a1_1")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


    # 模型预测
    points = np.c_[xx.ravel(), yy.ravel()]
    logits = model.predict(points, verbose=0)
    yhat = tf.argmax(logits, axis=1)

    # 绘制决策区域
    cmp = ListedColormap(['blue', 'orange', 'green', 'purple'])
    plt.contourf(xx, yy, yhat.numpy().reshape(xx.shape), cmap=cmp, alpha=0.3)

    # 绘制原始数据点
    for i in range(classes):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=cmp(i), label=f"class {i}", s=40)

    plt.title("Decision Boundary (Multiclass)")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.axis('equal')
    plt.show()
#应用make_blobs函数制作一个四类数据集
classes = 4
m = 100
centers = [[-5,2],[-2,-2],[1,2],[5,-2]]
std = 1.0
X_train,y_train =make_blobs(n_samples=m,centers=centers,cluster_std=std,random_state=30)

'''plt_mc(X_train,y_train,classes,centers,std=std)'''

'''# show classes in data set
print(f"unique classes {np.unique(y_train)}")
# show how classes are represented
print(f"class representation {y_train[:10]}")
# show shapes of our dataset
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")
'''

tf.random.set_seed(1234)
model = Sequential(
    [
    Dense(2,activation = 'relu',name = "Layer1"),
    Dense(4,activation = 'linear',name = "Layer2")
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#采用交叉熵损失函数，from_logits+TUre作为损失函数的参数
    optimizer=tf.keras.optimizers.Adam(0.01),#学习率0.01
)

model.fit(
    X_train,y_train,
    epochs=200#训练轮次
)


'''plt_cat_mc(X_train, y_train, model, classes)'''

#从第一层中提取权重
l1 = model.get_layer("Layer1")
W1,b1 = l1.get_weights()


# 绘制神经网络单层 ReLU 层输出特征空间（吴恩达课程 plt_layer_relu）
def plt_layer_relu(X, y, w1, b1, classes):
    z1 = np.dot(X, w1.T) + b1
    a1 = np.maximum(0, z1)
    cmp = ListedColormap(['blue', 'orange', 'green', 'purple'])
    plt.figure(figsize=(6, 4))
    for i in range(classes):
        plt.scatter(a1[y == i, 0], a1[y == i, 1], color=cmp(i), label=f"class {i}", s=40)
    plt.title("Layer 1 Output (ReLU Activated)")
    plt.xlabel("a1_0")
    plt.ylabel("a1_1")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

plt_layer_relu(X_train,y_train.reshape(-1,),W1,b1,classes)
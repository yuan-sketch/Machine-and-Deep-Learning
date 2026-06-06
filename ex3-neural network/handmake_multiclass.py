import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from autils import *
from lab_utils_softmax import plt_softmax
import numpy as np
np.set_printoptions(precision=2)

# ======================
# 修复在这里 ✅
# ======================
def my_softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

'''# 测试
z = np.array([1., 2., 3., 4.])
a = my_softmax(z)
atf = tf.nn.softmax(z)
print(f"my_softmax(z):         {a}")
print(f"tensorflow softmax(z): {atf}")

plt.close("all")
plt_softmax(my_softmax)  # 现在正常运行！'''


def display_errors(model, X, y):
    """
    显示模型预测错误的手写数字
    参数：
        model：训练好的TF模型
        X：特征数据 (5000,400)
        y：真实标签 (5000,1)
    """
    from matplotlib.colors import ListedColormap
    cmp = ListedColormap(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])

    # 模型预测
    predictions = model.predict(X, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = y.flatten()

    # 找出错误的索引
    errors_idx = np.where(pred_labels != true_labels)[0]

    if len(errors_idx) == 0:
        print("✅ 没有预测错误！")
        return

    print(f"❌ 错误总数：{len(errors_idx)}")

    # 最多显示 16 个错误
    n_display = min(16, len(errors_idx))
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    fig.suptitle("reality ↔ prediction", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i >= n_display:
            break
        idx = errors_idx[i]
        img = X[idx].reshape((20, 20)).T
        true_l = true_labels[idx]
        pred_l = pred_labels[idx]

        ax.imshow(img, cmap='gray')
        ax.set_title(f"{true_l} → {pred_l}", color='red')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_loss_tf(history):
    # 绘制 TensorFlow 模型训练损失曲线
    loss = history.history['loss']

    plt.figure(figsize=(6, 4))
    plt.plot(loss, label='Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Sparse Categorical Crossentropy)')
    plt.legend()
    plt.grid(True)
    plt.show()


def display_digit(x):
    """
    显示单张手写数字（20x20）
    参数 x：形状为 (400,) 的一维向量
    """
    # 重塑为 20x20 并转置（课程数据存储格式要求）
    img = x.reshape((20, 20)).T

    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()
X,y=load_data_all()
y[y == 10] = 0
# 吴恩达课程专用 widgvis 函数（修复报错）
def widgvis(fig):
    import matplotlib.widgets as widgets
    try:
        # 仅在交互式环境中调整控件位置
        for ax in fig.axes:
            for child in ax.get_children():
                if isinstance(child, widgets.Slider):
                    child.label.set_fontsize(9)
                    child.valtext.set_fontsize(9)
    except:
        pass

'''import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

# fig.tight_layout(pad=0.5)
widgvis(fig)
for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Display the label above the image
    ax.set_title(y[random_index, 0])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)
plt.show()
'''
tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ###
        tf.keras.Input(shape=(400,)),     # @REPLACE
        Dense(25, activation='relu', name = "L1"), # @REPLACE
        Dense(15, activation='relu',  name = "L2"), # @REPLACE
        Dense(10, activation='linear', name = "L3"),  # @REPLACE
        ### END CODE HERE ###
    ], name = "my_model"
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=40
)
'''plot_loss_tf(history)


image_of_two = X[1015]
display_digit(image_of_two)

prediction = model.predict(image_of_two.reshape(1,400))  # prediction

print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

prediction_p = tf.nn.softmax(prediction)

print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")
'''
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
widgvis(fig)
for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1, 400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)

    # Display the label above the image
    ax.set_title(f"{y[random_index, 0]},{yhat}", fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()

display_errors(model, X, y)
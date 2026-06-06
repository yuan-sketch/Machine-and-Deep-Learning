# 袁超
# 开发时间：2026/4/10 12:27
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
import matplotlib.pyplot as plt
from autils import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X,y=load_data()
print('X的第一个元素是：\n',X[0])
y[y == 10] = 0
filter_mask = (y == 0) | (y == 1)
X = X[filter_mask.ravel()]
y = y[filter_mask.ravel()]
def my_dense_v(A_in,W,b,g):
    Z = np.matmul(A_in,W)+b
    A_out = g(Z)
    return(A_out)

def my_sequential_v(X,W1,b1,W2,b2,W3,b3):
    A1 = my_dense_v(X,W1,b1,sigmoid)
    A2 = my_dense_v(A1,W2,b2,sigmoid)
    A3 = my_dense_v(A2,W3,b3,sigmoid)
    return(A3)

model =Sequential([
    Input(shape=(400,)),
    Dense(units=25, activation='sigmoid'),  # layer1
    Dense(units=15, activation='sigmoid'),                     # layer2
    Dense(units=1, activation='sigmoid')                      # layer3
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)
model.fit(X, y, epochs=20)
# ===================== 【3】获取初始权重 W1,b1,W2,b2,W3,b3 =====================
W1_tmp, b1_tmp = model.layers[0].get_weights()
W2_tmp, b2_tmp = model.layers[1].get_weights()
W3_tmp, b3_tmp = model.layers[2].get_weights()

# ===================== 【4】传给你自己的函数运行 =====================
Prediction = my_sequential_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
Yhat = (Prediction >= 0.5).astype(int)
print("predict a zero: ", Yhat[0], "predict a one: ", Yhat[500])

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    prediction = model.predict(X[random_index].reshape(1, 400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0

    # Display the label above the image
    ax.set_title(f"{y[random_index, 0]}, {Yhat[random_index, 0]}")
    ax.set_axis_off()
fig.suptitle("Label, Yhat", fontsize=16)
plt.show()
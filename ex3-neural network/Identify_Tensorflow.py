# 袁超
# 开发时间：2026/4/9 15:24
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#每个训练实例是一个20像素x20像素的灰度数字图像。
#每个像素由一个浮点数表示，表明该位置的灰度强度。
#20乘20的像素网格被 "展开（unrolled） "成一个400维的向量。
#每个训练实例都成为我们的数据矩阵X中的一个单行。
#这样我们就得到了一个1000 x 400的矩阵X，其中每一行都是一个手写数字图像的训练实例。

#加载数据
X,y=load_data()
print('X的第一个元素是：\n',X[0])
y[y == 10] = 0
filter_mask = (y == 0) | (y == 1)
X = X[filter_mask.ravel()]
y = y[filter_mask.ravel()]
#数据可视化

'''import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

m,n = X.shape

fig,axes = plt.subplots(8,8,figsize=(8,8))
fig.tight_layout(pad=0.5)

for i,ax in enumerate(axes.flat):
    #随机选择
    random_index = np.random.randint(m)
    #重画图像
    X_random_reshaped = X[random_index].reshape((20,20)).T
    #显示图像
    ax.imshow(X_random_reshaped,cmap='gray')
    #显示标题
    label = int(y[random_index].item())
    if label == 10:
        label = 0
    ax.set_title(label, fontsize=10)
    ax.set_axis_off()

plt.show()
'''
#模型表示
#该网络有三个密集层（25-15-1），并有sigmoid激活。
#由于图像的大小为20×20，因此我们有400的输入
#第1层。W1的形状是（400，25），b1的形状是（25，）。
#第2层。W2的形状是（25，15），b2的形状是：（15，）。
#第三层。W3的形状是（15，1），b3的形状是：（1，）。b是一维向量

model=Sequential(
    [

        tf.keras.Input(shape=(400,)),
        Dense(25, activation='sigmoid', name='layer1'),
        Dense(15, activation='sigmoid', name='layer2'),
        Dense(1 , activation='sigmoid', name='layer3')

    ],
    name="my_model"
)
model.summary()

L1_num_params = 400*25+25
L2_num_params = 25*15+15
L3_num_params = 15*1+15
[layer1,layer2,layer3] = model.layers

W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
'''print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")'''

#定义一个损失函数
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
model.fit(
    X,y,
    epochs=20
)

#检测一下概率准不准确
'''prediction = model.predict(X[0].reshape(1,400))
print(f"预测为0的概率是：{prediction}")
prediction = model.predict(X[500].reshape(1,400))
print(f"预测为1的概率是：{prediction}")'''

#比较一下64个样本的预测结果和标签是否匹配
import  warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
m,n = X.shape
fig,axes = plt.subplots(8,8,figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0,0.03,1,0.92])

for i,ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    X_random_reshaped = X[random_index].reshape((20,20)).T
    ax.imshow(X_random_reshaped,cmap='gray')
    prediction = model.predict(X[random_index].reshape(1,400))
    if prediction >=0.5:
        yhat = 1
    else:
        yhat = 0
    ax.set_title(f"{y[random_index,0]},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label,yhat",fontsize=16)
plt.show()
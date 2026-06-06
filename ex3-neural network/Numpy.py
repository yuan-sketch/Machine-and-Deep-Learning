# 袁超
# 开发时间：2026/4/9 14:27
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from lab_utils_common import dlc,sigmoid
from lab_coffee_utils import load_coffee_data,plt_roast,plt_prob,plt_layer,plt_network,plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
#查看数据情况
X,Y = load_coffee_data();
print(X.shape, Y.shape)
plt_roast(X,Y)

#数据归一化
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

def my_dense(a_in,W,b,g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units=W.shape[1]
    a_out=np.zeros(units)
    for j in range(units):
        w=W[:,j]
        z=np.dot(w,a_in)+b[j]
        a_out[j]=g(z)
    return(a_out)

#构建两层网络
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    return(a2)

#赋值权重和偏置值
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

#预测函数，模型输出是概率，需要和一个阈值进行比较，输入是一个矩阵X，行中又所有m的例子
def my_predict(X , W1 , b1, W2 , b2):
    m=X.shape[0]
    p=np.zeros((m,1))
    for i in range(m):
        p[i,0]=my_sequential(X[i],W1,b1,W2,b2)[0]
    return(p)
X_tst=np.array(
    [
        [200,13.9],
        [200,17]
    ]
)
X_tstn=norm_l(X_tst)
predictions=my_predict(X_tstn,W1_tmp,b1_tmp,W2_tmp,b2_tmp)

yhat=np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i]>=0.5:
        yhat[i]=1
    else:
        yhat[i]=0
print(f"结果是\n{yhat}")

#网络函数
netf=lambda x: my_predict(norm_l(x),W1_tmp,b1_tmp,W2_tmp,b2_tmp)
plt_network(X,Y,netf)


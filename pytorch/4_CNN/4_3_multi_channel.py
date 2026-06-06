# 袁超
# 开发时间：2026/5/8 16:13
import torch
from torch import nn
from my_utils_CNN import corr2d

#多输入通道
def corr2d_multi_in(X, K):
    #沿着X和K的第0维分别计算再相加
    res = corr2d(X[0,:,:],K[0,:,:])
    for i in range(1, X.shape[0]):
        res +=corr2d(X[i,:,:],K[i,:,:])
    return res

X = torch.tensor(
    [
        [[0,1,2],[3,4,5],[6,7,8]],
        [[1,2,3],[4,5,6],[7,8,9]]
    ],dtype=torch.float
)

K = torch.tensor(
    [
        [[0,1],[2,3]],
        [[1,2],[3,4]]
    ],dtype=torch.float
)

print(corr2d_multi_in(X, K))


#多输出通道
def corr2d_multi_in_out(X, K):
    #对K的第0维遍历，每次同输入X做互相关计算，所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X,k)for k in K])

K = torch.stack([K,K+1,K+2])

print(corr2d_multi_in_out(X, K))


#1*1卷积层
def corr2d_multi_in_out_11(X,K):
    #通道、高、宽
    c_i, h, w = X.shape
    #输出通道数
    c_o = K.shape[0]
    X = X.view(c_i,h*w)
    K = K.view(c_o,c_i)
    #全连接层的矩阵乘法
    Y = torch.mm(K,X)
    return Y.view(c_o,h,w)
X = torch.rand(3,3,3)
K = torch.rand(2,3,1,1)
Y1 = corr2d_multi_in_out(X,K)
Y2 = corr2d_multi_in_out_11(X,K)
print((Y1-Y2).norm().item()<1e-6)
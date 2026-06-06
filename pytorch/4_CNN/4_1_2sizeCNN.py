# 袁超
# 开发时间：2026/5/8 14:24
import torch
from torch import nn


#二维互相关运算
def corr2d(X, K):
    #卷积核的行，列数值
    h, w = K.shape
    #卷积结果的存放位置
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()
    return Y

X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
K = torch.tensor([[0,1],[2,3]])
print(corr2d(X,K))

#二维卷积层---输出=输入和卷积核做互相关运算+标量偏差
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x,self.weight)+self.bias

#图像中物体边缘检验

#首先创建一张6*8的图像，中间四列是黑（0），其余为白（1）
x = torch.ones(6, 8)
x[:,2:6] = 0

K = torch.tensor([[1,-1]])

y = corr2d(x,K.float())
print(y)

print('-'*100)

conv2d = Conv2D(kernel_size=(1, 2))
step = 30
lr =0.01
for i in range(step):
    Y_hat = conv2d(x)
    l = ((Y_hat - y)**2).sum()
    l.backward()
    #梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr* conv2d.bias.grad
    #梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i+1)%5 == 0:
        print('step %d, loss%.3f'%(i+1, l.item()))

print('weight = ',conv2d.weight.data)
print('bias = ',conv2d.bias.data)

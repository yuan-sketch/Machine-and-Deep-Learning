# 袁超
# 开发时间：2026/5/8 15:39
import torch
from torch import nn


#定义一个函数来计算卷积层，它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d,X):
    #(1,1)代表批量大小和通道数
    #(1,1)+(8,8)=(1,1,8,8)
    X = X.view((1,1)+X.shape)
    Y = conv2d(X)
    #排除不相关的前两维：批量和通道
    return Y.view(Y.shape[2:])
#两侧分别填充1行或1列
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=3, padding=1)

X = torch.rand(8,8)
print(comp_conv2d(conv2d,X).shape)
print("-"*100)
#使用高为5，宽为3的卷积核，在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(
    in_channels=1, out_channels=1,
    kernel_size=(5,3), padding=(2,1)
)

print(comp_conv2d(conv2d,X).shape)

print("-"*100)

#令高和宽上的步幅都是2
conv2d = nn.Conv2d(1,1,3,2,1)
print(comp_conv2d(conv2d,X).shape)

print("-"*100)

#高上步幅是3，宽上步幅是4
conv2d = nn.Conv2d(1,1,(3,5),(3,4),(0,1))
print(comp_conv2d(conv2d,X).shape)
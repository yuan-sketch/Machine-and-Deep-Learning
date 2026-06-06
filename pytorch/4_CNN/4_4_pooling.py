# 袁超
# 开发时间：2026/5/8 17:28
import torch
from torch import nn

#最大池化和平均池化
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    # 存储池化计算结果
    Y = torch.zeros(X.shape[0]-p_h+1, X.shape[1]-p_w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i: i+p_h,j: j+p_w].max()
            elif mode =='avg':
                Y[i,j] = X[i: i+p_h,j:j+p_w].mean()
        return Y


#填充和步幅
X = torch.arange(16, dtype=torch.float).view(1,1,4,4)
#默认情况下， MaxPool2d 实例里步幅和池化窗口形状相同。
pool2d = nn.MaxPool2d(3)
#可以手动指定步幅和填充。
pool2d = nn.MaxPool2d(3, padding=1,stride=2)
#指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅
pool2d = nn.MaxPool2d((2,4),padding=(1,2),stride=(2,3))

#多通道————池化层的输出通道数与输入通道数相等
X = torch.cat((X,X+1), dim=1)
pool2d = nn.MaxPool2d(3, padding=1,stride=2)
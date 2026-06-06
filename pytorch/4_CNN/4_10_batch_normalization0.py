# 袁超
# 开发时间：2026/5/11 15:05
import torch
from torch import nn
from my_utils_CNN import FlattenLayer,load_data_fashion_mnist,train_ch5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #判断当前模式是训练模式还是预测模式
    if not is_training:
        #如果是在预测模式下，直接使用传入的移动平均所得到得均值和方差
        X_hat = (X - moving_mean)/torch.sqrt(moving_var + eps)
    else:
        #前一层需要是全连接层或卷积层
        assert len(X.shape) in (2, 4)
        #全连接层
        if len(X.shape) == 2:
           #沿着纵向求均值，（1，特征个数）
           #逐特征求均值
           mean = X.mean(dim=0)
           var = ((X-mean)**2).mean(dim=0)
        else:
            #使用二维卷积层得情况，计算通道维上的均值和方差
            #保持X的形状
            mean = X.mean(dim=0, keepdim = True).mean(dim=2, keepdim=True).mean(dim=3,keepdim=True)
            var = ((X-mean)**2).mean(dim=0,keepdim=True).mean(dim=2, keepdim=True).mean(dim=3,keepdim=True)
            #可以优化为mean = X.mean(dim=(0,2,3),keepdim=True)
        #训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        #一阶指数平滑算法
        moving_mean = momentum * moving_mean + (1.0- momentum) * mean
        moving_var = momentum *moving_var + (1.0 - momentum) * var

    #拉伸和偏移
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        #全连接层
        if num_dims == 2:
            shape = (1, num_features)
        #卷积层
        else:
            shape = (1, num_features, 1, 1)

        #参与求梯度和迭代的拉伸和偏移参数，分别初始化为0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        #不参与求梯度和迭代的变量，全在内存上初始化为0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    def forward(self, X):
        #如果X不在显存上，将moving_mean和moving_var复制到X所在的显存上
        if self.moving_mean.device !=X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to((X.device))
        #保存更新过的moving_mean和moving_var
        #Module实例的training属性默认为True,调用.eval()后设为false
        Y, self.moving_mean, self.moving_var = batch_norm(
            self.training, X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps = 1e-5, momentum= 0.9
        )
        return  Y


#使用批量归一化层的LeNet
net = nn.Sequential(
    nn.Conv2d(1,6,5),
    BatchNorm(6, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    BatchNorm(16, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2,2),
    FlattenLayer(),
    nn.Linear(16*4*4, 120),
    BatchNorm(120, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr)
train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)
# 袁超
# 开发时间：2026/5/11 17:48
import torch
from torch import nn
from my_utils_CNN import FlattenLayer,load_data_fashion_mnist,train_ch5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#nn中BatchNorm1d用于全连接层，BatchNorm2d用于卷积层，都需要指定输入num_features
net = nn.Sequential(
    nn.Conv2d(1,6,5),
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(2,2),
    nn.Conv2d(6,16,5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2,2),
    FlattenLayer(),
    nn.Linear(16*4*4, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84,10)
)

batch_size = 256
trian_iter, test_iter = load_data_fashion_mnist(batch_size)
lr, num_epochs = 0.001,5
optimizer = torch.optim.Adam(net.parameters(),lr)
train_ch5(net,trian_iter,test_iter,batch_size,optimizer,device,num_epochs)
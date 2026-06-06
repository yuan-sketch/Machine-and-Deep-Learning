# 袁超
# 开发时间：2026/5/6 15:51
import torch
import numpy as np
from torch import nn
from my_utils import FlattenLayer,train_ch3,load_data_fashion_mnist

#1.定义模型参数
#使用包含两个隐藏层的多层感知机，其中两个隐藏层的输出个数都是256
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784,10,256,256
w1 = torch.tensor(
    np.random.normal(0,0.01,size=(num_inputs,num_hiddens1)),
    dtype=torch.float,
    requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
w2 = torch.tensor(
    np.random.normal(0, 0.01,size=(num_hiddens1,num_hiddens2)),
    dtype=torch.float,
    requires_grad=True)
b2 = torch.zeros(num_hiddens2,requires_grad=True)
w3 = torch.tensor(
    np.random.normal(0,0.01,size=(num_hiddens2,num_outputs)),
    dtype=torch.float,
    requires_grad=True)
b3 = torch.zeros(num_outputs,requires_grad=True)

drop_prob1, drop_prob2 = 0.2,0.5


params = [w1,b1,w2,b2,w3,b3]
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, num_outputs)
)
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

#训练并测试模型
num_epochs, batch_size,lr =5, 256, 0.1
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter, = load_data_fashion_mnist(batch_size)
train_ch3(net,train_iter,test_iter,loss,
          num_epochs,batch_size,None,None,optimizer)
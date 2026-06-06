# 袁超
# 开发时间：2026/4/29 16:46
import torch
import numpy as np
import torchvision

from torchvision import transforms
import d2l
from d2l import torch as d2l
import torch.nn as nn
from my_utils import load_data_fashion_mnist,evaluate_accuracy,train_ch3,FlattenLayer


num_inputs, num_outputs, num_hiddens = 784,10,256
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs)
)

for params in net.parameters():
    nn.init.normal_(params,mean=0,std=0.01)
#x训练模型
batch_size = 256
train_iter,test_iter = load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
num_epochs = 5
train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)
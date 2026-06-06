# 袁超
# 开发时间：2026/5/6 14:17
import torch
import numpy as np
import torchvision
from torchvision import transforms
from my_utils import load_data_fashion_mnist,train_ch3

def dropout(X, drop_prob):
    X = X.float()
    assert 0<= drop_prob <=1
    keep_prob = 1- drop_prob
    #这种情况下把全部元素都去掉
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X/keep_prob

X = torch.arange(16).view(2,8)

print(dropout(X,1))

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

params = [w1,b1,w2,b2,w3,b3]

#2.定义模型
drop_prob1, drop_prob2 = 0.2,0.5
def net(X, is_training=True):
    X = X.view(-1,num_inputs)
    H1 = (torch.matmul(X,w1) + b1).relu()
    #只在模型训练时使用丢弃法
    if is_training:
        #在第一层全连接后添加丢弃层
        H1 = dropout(H1,drop_prob1)
    H2 = (torch.matmul(H1,w2) + b2).relu()
    if is_training:
        #在第二层全连接后添加丢弃层
        H2 = dropout(H2, drop_prob2)
    return torch.matmul(H2,w3) + b3

def evaluate_accuracy( net,data_iter):
    """
    function：
    计算多分类模型预测结果的准确率
    Parameters:
    data_iter - 样本划分为最小批的结果
    net - 定义的网络
    Returns:
    准确率计算结果
    """
    acc_sum, n = 0.0,0
    for X, y in data_iter:
        if isinstance(net,torch.nn.Module):
            #评估模式，这会关闭dropout
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            #改回训练模式
            net.train()
        else:
            if('is_training'in net.__code__.co_varnames):
                #将is_training设置成False
                acc_sum +=(net(X,is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n

#3.训练和测试模型
num_epochs, lr, batch_size = 10,0.1,256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter, = load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter,test_iter, loss,
          num_epochs, batch_size, params, lr)

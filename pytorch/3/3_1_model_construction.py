# 袁超
# 开发时间：2026/5/7 16:13
import torch
from torch import nn
import collections

#1.继承Module类来构造模型
class MLP(nn.Module):
    #声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self,**kwargs):
        #调用MLP父类Module的构造函数来进行初始化
        #这样在构造实例时还可以指定其他函数参数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784,256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256,10)
    def forward(self,x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.randn(2, 784)
net = MLP()
print(net(X))

#2.Sequential类继承自MOdule类
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        #如果传入的时一个OrderedDict
        if len(args)==1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].item():
                #add_module方法会将module添加
                #将self._module(一个OrderedDict)
                self.add_module(key, module)
        #传入的是一些Module
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input



net = MySequential(
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
#print(net)
#print(net(X))


#3.构造更复杂的模型
class FancyMLP(nn.Module):
    def __init__(self,**kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        #不可训练参数
        self.rand_weight = torch.rand((20,20), requires_grad= False)
        self.linear = nn.Linear(20,20)
    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        x = self.linear(x)
        #控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /=2
        if x.norm().item() <0.8:
            x *=10
        return x.sum()


X = torch.rand(2, 20)
net = FancyMLP()
#print(net)
#print(net(X))

#可以嵌套调用
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40,30), nn.ReLU())
    def forward(self, x):
        return self.net(x)

net = nn.Sequential(
    NestMLP(),
    nn.Linear(30,20),
    FancyMLP()
)
X = torch.rand(2,40)
print(net)
print(net(X))
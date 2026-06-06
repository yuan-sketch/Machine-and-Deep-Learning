# 袁超
# 开发时间：2026/5/7 18:06
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(20,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
#pytorch已经默认初始化
print(net)
X = torch.rand(2, 20)
Y = net(X)

print('-'*40,'第0层的参数、参数尺寸、类型','-'*40)
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))

param = dict(net[0].named_parameters())#将第0层的所有参数转换成一个字典，方便后续索引

print(type(param['weight']),
      param['weight'].size(),
      type(param['bias']),
      param['bias'].size()
      )
print('-'*100)
#初始化模型参数---init
from torch.nn import init
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param,mean=0,std=0.01)
        print(name,'\n', param.data)
    if'bias' in name:
        init.constant_(param, val=0)
        print(name,'\n', param.data)

print("-"*100)
#对某个特定参数进行初始化
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

print('-'*100)

#自定义初始化参数
#令权重有一半的概率初始化为0，有另一半的概率初始化为【-10，-5】和【5，10】两个区间里均匀分布的随机数
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10,10)
        tensor *=(tensor.abs() >=5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name,'\n', param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data +=1
        print(name,'\n', param.data)

print('-'*100)
#共享模型参数
#Module 类的 forward 函数里多次调用同一个层。此外，如果我们传入 Sequential 的模块是同一个 Module实例的话参数也是共享的
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name,param in net.named_parameters():
    init.constant_(param,val=3)
    print(name, param.data)

#因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的
x = torch.ones(1,1)
y = net(x).sum()
print(y)
y.backward()
#单次梯度3，两次6
print(net[0].weight.grad)



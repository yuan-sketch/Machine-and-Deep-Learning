# 袁超
# 开发时间：2026/5/7 19:36

import torch
from torch import nn



#不含模型参数的自定义层
class CentereLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CentereLayer, self).__init__(**kwargs)
    def forward(self, x):
            return x - x.mean()

layer = CentereLayer()
print(layer(torch.tensor([1,2,3,4,5],dtype=torch.float)))

net = nn.Sequential(
    nn.Linear(8, 128),
    CentereLayer()
)
print(net(torch.rand(4, 8)).mean().item())            #生成4个样本，每个样本8个特征（4行8列）
print('-'*100)
#含模型参数的自定义层
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn((4,4)))for i in range(3)]
        )
        self.params.append(nn.Parameter(torch.randn(4,1)))
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyDense()
print(net)
print('-'*100)

class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense,self).__init__()
        self.params = nn.ParameterDict(
            {
                'linear1':nn.Parameter(torch.randn(4,4)),
                'linear2':nn.Parameter(torch.randn(4,1))
            }
        )
        #新增
        self.params.update(
            {
                'linear3':nn.Parameter(torch.randn(4,2))
            }
        )

    def forward(self,x,choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

print('-'*100)

x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))

print('-'*100)

net = nn.Sequential(
    MyDictDense(),
    MyDense()
)

print(net)
print(net(x))


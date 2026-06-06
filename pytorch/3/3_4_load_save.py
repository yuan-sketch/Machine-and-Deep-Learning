# 袁超
# 开发时间：2026/5/8 13:28
import torch
from torch import nn


#读写tensor
x = torch.ones(3)
torch.save(x, '../x.pt')

x2 = torch.load('../x.pt')
print(x2)

#读写模型的参数
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()

torch.save(net, '../model.bin')

net2 = torch.load('../model.bin')

Y2 = net2(x)
Y = net(x)
print(Y2 == Y)
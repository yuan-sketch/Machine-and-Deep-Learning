# 袁超
# 开发时间：2026/4/27 18:10
import torch
w = torch.tensor(2)
b = torch.tensor(3)
x = torch.tensor(1)
y = x * w + b
w = torch.randn(3,2,requires_grad=True)
b = torch.randn(2,requires_grad=True)
x = torch.randn(1,3)
y = x @ w + b


import torch.nn.functional as F
y = F.relu(y)

Y = torch.randn(1,2)
loss = F.cross_entropy(y,Y)           #计算交叉熵损失


loss.backward()                       #反向传播
w = w -0.0001 * w.grad
print(loss.grad_fn)
print(y.grad_fn)
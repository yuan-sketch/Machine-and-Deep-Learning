# 袁超
# 开发时间：2026/5/21 13:59
import time
import numpy as np
import d2lzh6 as d2l
import torch.nn as nn
import torch.utils.data

def get_data_ch7():
    # 数据读取以\t分隔开，处理缺失值
    data = np.genfromtxt('data/airfoil_self_noise.dat',delimiter='\t')
    # 特征标准化处理
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32),torch.tensor(data[:1500, -1],dtype=torch.float32)

features, labels = get_data_ch7()
def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams,
                      features, labels, batch_size=10, num_epochs=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    # 使用**前缀，将输入的参数存放在字典里
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)
    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2
    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size, shuffle=True
    )
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致，因为loss中除了2
            l = loss(net(X).view(-1), y) / 2
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch'%(ls[-1], time.time()-start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()
from torch import optim
train_pytorch_ch7(optim.SGD, {'lr':0.05}, features, labels, 10)
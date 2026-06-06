# 袁超
# 开发时间：2026/5/21 12:03
import numpy as np
import torch
import d2lzh6 as d2l
import time
def get_data_ch7():
    # 数据读取以\t分隔开，处理缺失值
    data = np.genfromtxt('data/airfoil_self_noise.dat',delimiter='\t')
    # 特征标准化处理
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32),torch.tensor(data[:1500, -1],dtype=torch.float32)

features, labels = get_data_ch7()
print(labels,'\n', features[:20],'\n',features.shape)

def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data

def train_ch7(optimizer_fn, states, hyperparams,
              features, labels, batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = d2l.linreg, d2l.squared_loss
    w = torch.nn.Parameter(
        torch.tensor(
            np.random.normal(0, 0.01, size=(features.shape[1], 1)),
            dtype=torch.float32
        ),requires_grad=True
    )
    b = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.float32),
        requires_grad=True
    )
    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()
    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size,
        shuffle=True
    )
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X,y) in enumerate(data_iter):
            # 平均损失
            l = loss(net(X, w, b), y).mean()
            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            # 迭代参数
            optimizer_fn([w, b], states, hyperparams)
            # 每100个样本记录下训练误差
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())

    print('loss: %f, %f sec per epoch'%(ls[-1], time.time()-start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()
def train_sgd(lr, batch_size, num_epochs):
    train_ch7(sgd, None, {'lr':lr}, features,
              labels, batch_size, num_epochs)

# 批量是1500时，梯度下降
train_sgd(1, 1500, 6)
# 批量是1时， 随机梯度下降
train_sgd(0.005,1,2)
# 批量是10时，小批量随即下降
train_sgd(0.05, 10, 2)
# 袁超
# 开发时间：2026/5/5 22:36
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
# 生成数据集（高维线性回归，用于演示过拟合 + 权重衰减）
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

train_features = features[:n_train, :]
test_features = features[n_train:, :]
train_labels = labels[:n_train]
test_labels = labels[n_train:]

# 1. 初始化模型参数
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 2. 定义L2范数惩罚项（权重衰减）
def l2_penalty(w):
    return (w ** 2).sum() / 2

# 3. 定义模型和损失函数（纯 PyTorch，删掉 d2l.mxnet）
def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 4. 优化器（纯 PyTorch 实现 SGD）
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 5. 绘图函数
def semilogy(x_vals, y_vals, x_label, y_label,
             x2_vals=None, y2_vals=None, legend=None, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

# 6. 训练 + 测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = linreg, squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())

# 7. 测试：不使用正则化（lambd=0），会出现严重过拟合
#fit_and_plot(lambd=0)

#使用权重衰减
#fit_and_plot(lambd=3)

def fit_and_plot_pytorch(wd):
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0,std=1)
    nn.init.normal_(net.bias, mean=0,std=1)
    #对权重参数衰减
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr,
                                  weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias],lr=lr)
    train_ls, test_ls = [],[]
    for _ in range(num_epochs):
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y.reshape(y_hat.shape)).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            #对两个optomizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features),
                             train_labels.reshape(-1,1)).mean().item())
        test_ls.append(loss(net(test_features),
                            test_labels.reshape(-1,1)).mean().item())
    semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',
             range(1,num_epochs+1),test_ls,['train','tset'])
    print('L2 norm of w:',net.weight.data.norm().item())

fit_and_plot_pytorch(0)

fit_and_plot_pytorch(3)
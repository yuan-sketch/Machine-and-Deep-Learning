# 袁超
# 开发时间：2026/5/5 14:06
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

# 1.生成数据集y=1.2x-3.4x^2+5.6x^3+5+随机误差
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((
    features,
    torch.pow(features, 2),
    torch.pow(features, 3)), 1)
labels = (
        true_w[0] * poly_features[:, 0] +
        true_w[1] * poly_features[:, 1] +
        true_w[2] * poly_features[:, 2] +
        true_b
)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

print(features[:2], poly_features[:2], labels[:2])


# 2.定义、训练和测试模型
def semilogy(x_vals, y_vals, x_label, y_label,
             x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    # 完全去掉 d2l，只用原生 matplotlib
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)  # 对数坐标

    # 绘制测试集曲线
    if x2_vals is not None and y2_vals is not None:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

    plt.show()  # 必须加，否则图不显示


num_epochs, loss = 100, nn.MSELoss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    # 定义线性模型
    net = nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])

    # 构造数据加载器
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []

    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()  # 清空梯度
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数

        # 记录训练 & 测试损失
        train_ls.append(loss(net(train_features), train_labels.view(-1, 1)).item())
        test_ls.append(loss(net(test_features), test_labels.view(-1, 1)).item())

    print('final epoch: train loss', train_ls[-1],
          'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)


# 开始训练
fit_and_plot(poly_features[:n_train, :],
             poly_features[n_train:, :],
             labels[:n_train],
             labels[n_train:])
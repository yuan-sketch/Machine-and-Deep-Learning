import torch
import numpy as np
import torchvision
from torchvision import transforms

batch_size = 256


# 加载本地数据
def load_data_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data/FashionMNIST", train=True, transform=transform, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data/FashionMNIST", train=False, transform=transform, download=False)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter


# 评估准确率
def evaluate_accuracy(net, data_iter):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    if optimizer is None:
        optimizer = torch.optim.SGD(params, lr=lr)

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(net, test_iter)
        print(
            f'epoch {epoch + 1}, loss {train_l_sum / n:.4f}, train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}')


# ======================
# 🔥 关键：把 net 写成类（继承 nn.Module）
# ======================
class MLP(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(MLP, self).__init__()
        self.W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float,
                               requires_grad=True)
        self.b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
        self.W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float,
                               requires_grad=True)
        self.b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

    def forward(self, X):
        X = X.view(-1, 784)
        H = torch.max(X @ self.W1 + self.b1, torch.tensor(0.0))
        return H @ self.W2 + self.b2


# 初始化模型
num_inputs, num_outputs, num_hiddens = 784, 10, 256
net = MLP(num_inputs, num_hiddens, num_outputs)
params = [net.W1, net.b1, net.W2, net.b2]

# 损失
loss = torch.nn.CrossEntropyLoss()

# 加载数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 训练
num_epochs, lr = 5, 0.1
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
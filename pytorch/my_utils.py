# 袁超
# 开发时间：2026/4/29 17:19
import torch
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy(), cmap='gray')
        f.set_title(lbl)
        f.axis('off')
    plt.show()

def get_fashion_mnist_labels(labels):
    text_labels = ['t_shirt', 'trouser', 'pullover','dress', 'coat', 'sandal',
                   'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in labels]


def load_data_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data/FashionMNIST", train=True, transform=transform, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data/FashionMNIST", train=False, transform=transform, download=False)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter


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

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)


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

# 袁超
# 开发时间：2026/5/8 16:23
import torch
import time
import sys
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def corr2d(X, K):
    """
    计算二维互相关运算
    X: 输入二维张量 (H, W)
    K: 卷积核 (h, w)
    返回: 输出张量 (H-h+1, W-w+1)
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1,
                      X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y



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


def load_data_fashion_mnist(batch_size, resize=None):
    """
    function：
    将fashion mnist数据集划分为小批量样本
    Parameters:
    batch_size - 小批量样本的大小(int)
    resize - 对图像的维度进行扩大
    Returns:
    train_iter - 训练集样本划分为最小批的结果
    test_iter - 测试集样本划分为最小批的结果
    Modify:添加图像维度变化
    """
    # 存储图像处理流程
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='D:\桌面\吴恩达机器学习笔记等资源\code\pytorch\data/FashionMNIST',
        train=True,
        download=False,
        transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='D:\桌面\吴恩达机器学习笔记等资源\code\pytorch\data/FashionMNIST',
        train=False,
        download=False,
        transform=transform
    )
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_iter,test_iter



def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    net.train()
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


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    """
    function：
    利用softmax回归模型对图像进行分类识别
    学习率采用0.001，训练算法使用Adam算法，损失函数使用交叉熵损失函数。
    小结：
    Parameters:
    net - 定义的网络
    train_iter - 训练集样本划分为最小批的结果
    test_iter - 测试集样本划分为最小批的结果
    num_epochs - 迭代次数
    batch_size - 最小批大小
    optimizer - 优化器
    device - 指定计算在GPU或者CPU上进行
    Returns:
    """
    net = net.to(device)
    print("training on",device)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count = 0.0,0.0,0,0
        start = time.time()
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            #梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum +=(y_hat.argmax(dim=1) == y).sum().cpu().item()
            n +=y.shape[0]
            batch_count +=1
        test_acc = evaluate_accuracy(test_iter, net,device=device)
        print('epoch %d, loss %f, train acc %.3f,test acc %.3f,\time %.1f sec'
              %(epoch+1, train_l_sum/batch_count, train_acc_sum/n,  test_acc, time.time()-start ))


class GlobalAvgPool2d(nn.Module):
    #全局平均池化将窗口形设置成高和宽
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


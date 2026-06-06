# 袁超
# 开发时间：2026/5/22 15:34
import time

import torchvision.datasets
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from PIL import Image
import d2lzh8 as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys

def show_image(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

all_imgs = torchvision.datasets.CIFAR10(
    train=True, root='data/CIFAR10',download=False
)
show_image([all_imgs[i][0] for i in range(32)],
           4, 8, scale=0.8)
d2l.plt.show()

# 用ToTensor将小批量图像转换成PyTorch需要的格式，形状为(批量大小，通道数，高，宽)
flip_aug = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ]
)
no_aug = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)

# 定义函数，读取图像并应用图像增广
num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size, root='data/CIFAR10'):
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=is_train, transform=augs, download=False
    )
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=is_train, num_workers=num_workers
    )

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    print("training on:", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0,0.0,0
        start = time.time()
        net.train()
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        net.eval()
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'%
              (epoch+1,train_l_sum/batch_count, train_acc_sum/n,test_acc, time.time()-start))

# 使用图像增广训练模型
def train_with_data_aug(train_augs, test_augs, lr = 0.001):
    batch_size, net = 256, d2l.resnet18(10)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss,optimizer,device,num_epochs=10)

train_with_data_aug(flip_aug, no_aug)


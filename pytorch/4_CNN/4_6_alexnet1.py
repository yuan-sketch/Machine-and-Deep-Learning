# 袁超
# 开发时间：2026/5/9 16:01

import sys
import time
import torch
import torchvision.datasets
from torch import nn, optim
from torchvision import transforms

# ====================== 强制验证GPU状态 ======================
print("=" * 60)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"最终使用的训练设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
print("=" * 60)


# ====================== 轻量化AlexNet（适配MX450） ======================
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=0),  # 从96减到64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),  # 从256减到192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),  # 从384减到256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 从384减到256
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 从256减到128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 2048),  # 从4096减到2048
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),  # 从4096减到1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, imag):
        feature = self.conv(imag)
        output = self.fc(feature.view(imag.shape[0], -1))
        return output


# ====================== 精度评估函数（优化版） ======================
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


# ====================== 训练函数（混合精度加速版） ======================
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    loss = nn.CrossEntropyLoss()

    # 开启混合精度训练（MX450必备，提速30%+，减少显存占用）
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        start = time.time()

        for X, y in train_iter:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                y_hat = net(X)
                l = loss(y_hat, y)

            optimizer.zero_grad()
            # 混合精度反向传播
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1

        test_acc = evaluate_accuracy(test_iter, net, device)
        print(
            f'epoch {epoch + 1}, loss {train_l_sum / batch_count:.4f}, train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}, time {time.time() - start:.1f} sec')


# ====================== 数据加载函数（优化版） ======================
def load_data_fashion_mnist(batch_size, resize=None):
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    data_root = r'D:\桌面\吴恩达机器学习笔记等资源\code\pytorch\data\FashionMNIST'

    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=False,
        transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=False,
        transform=transform
    )

    num_workers = 0 if sys.platform.startswith('win') else 4

    train_iter = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_iter, test_iter


# ====================== 主程序执行（MX450专属参数） ======================
if __name__ == '__main__':
    net = AlexNet()
    print(net)

    # MX450 2GB显存最佳batch_size：64（如果还是显存不足，降到32）
    batch_size = 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

    # 学习率稍微调大一点，配合小batch_size
    lr, num_epochs = 0.0015, 5
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 开始训练
    train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
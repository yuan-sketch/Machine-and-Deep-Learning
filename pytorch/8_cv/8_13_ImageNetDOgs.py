# 袁超
# 开发时间：2026/6/6 12:15
import d2lzh8 as d2l
import torch
import torchvision
import os
from torch import nn
import numpy as np
data_dir = 'data/kaggle_dogs/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 整理数据集
def reorg_dog_data(data_dir, valid_ratio):
    # 读取训练数据标签
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    # 整理训练数据集
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    # 整理测试数据集
    d2l.reorg_test(data_dir)

batch_size = 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# 图像增广
transform_train = torchvision.transforms.Compose(
    [
        # 随机对图像裁剪出面积为原图像0.08-1倍、宽和高之比在3/4-4/3的像素
        # 再缩放为高和宽均为224像素的新图像
        torchvision.transforms.RandomResizedCrop(
            224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)
        ),
        torchvision.transforms.RandomHorizontalFlip(),
        # 随机变化亮度、对比度和饱和度
        torchvision.transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4
        ),
        torchvision.transforms.ToTensor(),
        # 对图像的每个通道做标准化
        torchvision.transforms.Normalize(
            [0.485,0.456,0.406], [0.229,0.224,0.225]
        )
    ]
)
# 测试时，使用确定性的图像预处理操作
transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        # 将图像中央的高和宽均为224的正方形区域裁剪出来
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485,0.456,0.406],[0.229,0.224,0.225]
        )
    ]
)

# 读取数据集
# 创建 ImageFolder 实例来读取整理后的含原始图像文件的数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train
)for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test
)for folder in ['valid', 'test']]

# 创建DataLoader实例
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)
]
valid_iter = (
    torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
)
test_iter = (
    torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)
)

# 定义模型
def get_net(device):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(
        pretrained=True
    )
    # 定义新的输出网络，输出类别个数是120
    finetune_net.output_new = nn.Sequential(
        nn.Linear(1000,256),
        nn.ReLU(),
        nn.Linear(256,120)
    )
    # 将模型参数分配到显存上
    finetune_net = finetune_net.to(device)
    for param in finetune_net.parameters():
        param.requires_grad = True
    return finetune_net

loss = nn.CrossEntropyLoss()
def evaluate_loss(data_iter, net, devices):
    l_sum, n =0.0,0
    for features, labels in data_iter:
        features, labels = features.to(devices), labels.to(devices)
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum = l.sum()
        n += labels.numel()
    return l_sum/n

# 训练模型
devices, num_epochs, lr, wd = device, 5, 0.01, 1e-4
net = get_net(devices)
optimizer = torch.optim.SGD(
    net.parameters(), lr, momentum=0.9, weight_decay=wd
)
d2l.train(train_iter, valid_iter, net, loss, optimizer, device, num_epochs)



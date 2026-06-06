# 袁超
# 开发时间：2026/6/3 11:40
import collections
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
import d2lzh8 as d2l
data_dir = 'data/kaggle_cifar10/'

def read_csv_labels(fname):
    """
    读取文件，返回字典，返回图片名称到标签之间的映射
    """
    with open(fname, 'r') as f:
        #跳过表头
        lines = f.readlines()[1: ]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((id, label) for id, label in tokens))
labels = read_csv_labels(
    os.path.join(data_dir, 'trainLabels.csv')
)
print('# 训练样例:',len(labels))
print('# 类别:',len(set(labels.values())))

# 定义reorg_train_valid函数从原始数据集中切分出验证集
# valid_ratio 是验证集样本数与原始训练集样本集之比
# 训练集放在train_valid_test/train下，验证集放在train_valid_test/valid下
def copyfile(filename, target_dir):
    """
    将文件复制到目标路径下
    """
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    # 训练集中样本数最少的类别包含的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每一类的样本数
    n_valid_per_label = max(1, math.floor(n*valid_ratio))
    label_count = {}
    for train_file in os.listdir(
        os.path.join(data_dir, 'train')
    ):
        # 为每个训练集样本匹配类别
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        # 以类别名为文件夹名称保存数据
        # 训练集与验证集合并
        copyfile(fname, os.path.join(
            data_dir, 'train_valid_test', 'train_valid', label
        ))
        if label not in label_count or label_count[label] < n_valid_per_label:
            # 仅包含验证集
            copyfile(fname, os.path.join(
                data_dir, 'train_valid_test', 'valid', label
            ))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            # 仅包含训练集
            copyfile(fname, os.path.join(
                data_dir, 'train_valid_test', 'train', label
            ))
    return n_valid_per_label

# 用reorg_test函数整理测试集
def reorg_test(data_dir):
    # 测试集数据整理，类别名为unknown
    for test_file in os.listdir(os.path.join(data_dir,'test','test')):
        copyfile(
            os.path.join(data_dir, 'test', 'test', test_file),
            os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')
        )

def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(
        os.path.join(data_dir, 'trainLabels.csv')
    )
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

# 用百分之10的训练样本作为调参使用的验证集
batch_size = 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)

# 为应对过拟合，使用图片增广
transform_train = torchvision.transforms.Compose(
    [
        # 使图像放大成宽和高均为40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪为原图像0.64-1倍的小正方形
        # 再缩放为高和宽各为32像素的正方形
        torchvision.transforms.RandomResizedCrop(
            32, scale=(0.64,1), ratio=(1.0,1.0)
        ),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 对图像的每个通道标准化
        torchvision.transforms.Normalize(
            [0.4914, 0.4822, 0.4465],
            [0.2023, 0.1994, 0.2010]
        )
    ]
)

# 测试时，仅对图像做标准化
transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.4914, 0.4822, 0.4465],
            [0.2023, 0.1994, 0.2010]
        )
    ]
)


# 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform = transform_train
) for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test
) for folder in ['valid', 'test']]

# 在DataLoader中指明定义好的图像增广操作
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True
    ) for dataset in (train_ds, train_valid_ds)
]
valid_iter = torch.utils.data.DataLoader(
    valid_ds, batch_size, shuffle=False, drop_last=True
)
test_iter = torch.utils.data.DataLoader(
    test_ds, batch_size, shuffle=False, drop_last=True
)
# 定义模型
# 使用ResNet-18模型对数据样本进行训练
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net
net = get_net()
loss = nn.CrossEntropyLoss()

# 训练模型
device, num_epochs, lr, wd = 'cuda', 5, 0.1, 5e-4
optimizer = torch.optim.SGD(
    net.parameters(), lr, momentum=0.9, weight_decay=wd
)
d2l.train(train_iter,test_iter,net,loss,optimizer,device,num_epochs)

# 袁超
# 开发时间：2026/5/22 19:26
import torch
from torch import nn,optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import sys
import d2lzh8 as d2l
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

data_dir = 'data'
os.listdir(os.path.join(data_dir, 'hotdog'))
train_imgs = ImageFolder(
    os.path.join(data_dir, 'hotdog/train')
)
test_imgs = ImageFolder(
    os.path.join(data_dir, 'hotdog/test')
)

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]

d2l.show_images(hotdogs+not_hotdogs, 2, 8, scale = 1.4)
d2l.plt.show()

# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)
train_augs = transforms.Compose(
    [
        # 统一大小
        transforms.RandomResizedCrop(size=224),
        # 水平翻转
        transforms.RandomHorizontalFlip(),
        # 0-1之间
        transforms.ToTensor(),
        # 标准化
        normalize
    ]
)

test_augs = transforms.Compose(
    [
        transforms.Resize(size=256),
        # 中心裁剪
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ]
)

# 定义和初始化模型
# 使用再Imagenet上预训练的ResNet-18作为源模型
# 指定pretrained=True自动下载并加载预训练模型参数
pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net.fc)

outputs_params = list(map(id, pretrained_net.fc.parameters()))
features_params = filter(
    lambda p: id(p) not in outputs_params,
    pretrained_net.parameters()
)
lr = 0.01
optimizer = optim.SGD(
    [
        {'params': features_params},
        {'params': pretrained_net.fc.parameters(), 'lr':lr*10}
    ],
    lr = lr,
    weight_decay=0.001
)
# 定义一个微调模型
def train_fine_tining(net, optimizer, batch_size = 128, num_epochs=5):
    net = net.to(device)
    train_iter = DataLoader(
        ImageFolder(
            os.path.join(data_dir, 'hotdog/train'),
            transform=train_augs
        ),
        batch_size,shuffle=True
    )
    test_iter = DataLoader(
        ImageFolder(
            os.path.join(data_dir, 'hotdog/test'),
            transform=test_augs
        ),
        batch_size
    )
    loss = nn.CrossEntropyLoss()
    d2l.custom_train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# train_fine_tining(pretrained_net,optimizer,128,10)
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(
    scratch_net.parameters(), lr = lr, weight_decay=0.001
)
train_fine_tining(scratch_net, optimizer)
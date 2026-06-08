# 袁超
# 开发时间：2026/6/8 14:55

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
import torch.nn as nn
import torch.utils.data as data
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 读取数据集
# 读取数据路径
def make_dataset(root):
    imgs = []
    # 计算共有多少张原始图片
    n = len(os.listdir(root))//2
    for i in range(n):
        # 找到名字为00i.png的路径
        img = os.path.join(root, '%03d.png'%i)
        # 找到名字为00i_mask.png的路径
        mask = os.path.join(root, '%03d_mask.png'%i)
        # 添加到列表
        imgs.append((img, mask))
    return imgs

# 数据增强
x_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5,0.5,0.5],[0.5,0.5,0.5]
        )
    ]
)
# mask只转为TOtensor
y_transforms = transforms.ToTensor()


# 图片预处理
# 定义LiverDataset类来读取图片及其标签图片，并进行相应的预处理
class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y
    def __len__(self):
        return len(self.imgs)

# 定义模型
# U-Net 模型中的双卷积网络结构
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        # 特征图大小不变
        self.conv1 = DoubleConv(in_channels, 64)
        # 特征图长宽减半
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 长宽翻倍， 通道数减半
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        # 通道维拼接
        merge6 = torch.cat([up_6, c4], dim=1)         # batch, channel, h, w
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

# 定义训练函数
def train_model(model, loss, optimizer, dataloaders, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1 , num_epochs))
        print('-'*20)
        dt_size = len(dataloaders.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataloaders:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            epoch_loss += l.item()
            if step % 200 == 0:
                print('%d/%d, train_loss: %.3f'%(step, (dt_size-1)//dataloaders.batch_size+1, l.item()))
        print('epoch %d loss:%.3f' %(epoch, epoch_loss))
    return model
if __name__ == '__main__':
    batch_size = 1
    liver_dataset = LiverDataset(
        'data/liver/train',
        transform=x_transforms,
        target_transform=y_transforms
    )

    dataloaders = DataLoader(liver_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4)
    # 输入图像通道数3，标签图像通道数1
    net = Unet(3, 1).to(device)
    # 采用BCEloss损失函数，该损失函数用于图片多标签分类
    loss = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    model = train_model(net, loss, optimizer, dataloaders)


# 袁超
# 开发时间：2026/5/22 14:23
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from PIL import Image
import d2lzh8 as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


d2l.set_figsize()
img = Image.open('data/cat1.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

# 定义绘图函数
def show_image(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

# 定义辅助函数apply
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows*num_cols)]
    show_image(Y, num_rows, num_cols, scale)

# 八张图以p=0.5的概率左右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
d2l.plt.show()
# 上下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
d2l.plt.show()

# 八张图；每次随机裁剪一块面积为原面积10%-100%的区域
# 该区域的宽和高之比随机取自0.5-2
# 再将该区域的宽和高分别缩放到200像素
shape_aug = torchvision.transforms.RandomResizedCrop(
    200, scale=(0.1, 1), ratio=(0.5, 2)
)
apply(img, shape_aug)
d2l.plt.show()

# 变化颜色
# 亮度（brightness),对比度(contrast),饱和度(saturation),色调(hue)
# 亮度变化为原来的50%-150%
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
d2l.plt.show()

apply(img, torchvision.transforms.ColorJitter(hue=0.5))
d2l.plt.show()

# 同时设置随机改变图像的亮度，对比度，饱和度，色调
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5,
    saturation=0.5, hue=0.5
)
apply(img, color_aug)
d2l.plt.show()

# 叠加多个图像增广
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,
    shape_aug
])
apply(img,augs)
d2l.plt.show()
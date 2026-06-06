# 袁超
# 开发时间：2026/6/1 20:07
# 功能：PASCAL VOC2012语义分割数据集的读取、预处理与数据加载

import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import d2lzh8 as d2l


def read_voc_images(root='data/VOCdevkit/VOC2012',
                    is_train=True, max_num=None):
    """读取PASCAL VOC2012数据集的图像和对应的语义分割标签"""
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt'
    )
    with open(txt_fname, 'r') as f:
        images = f.read().split()

    if max_num is not None:
        images = images[:min(max_num, len(images))]

    features, labels = [None] * len(images), [None] * len(images)

    for i, fname in tqdm(enumerate(images)):
        features[i] = Image.open(
            '%s/JPEGImages/%s.jpg' % (root, fname)
        ).convert('RGB')
        # 已修正原代码拼写错误：SegmentationCLass -> SegmentationClass
        labels[i] = Image.open(
            '%s/SegmentationClass/%s.png' % (root, fname)
        ).convert('RGB')

    return features, labels


voc_dir = 'data/VOCdevkit/VOC2012'
train_features, train_labels = read_voc_images(voc_dir, max_num=100)

# 可视化前5张图像及其标签
n = 5
imgs = train_features[:n] + train_labels[:n]
d2l.show_images(imgs, 2, n)
d2l.plt.show()

# VOC2012语义分割标签颜色映射表
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128],
                [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128],
                [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# VOC2012语义分割类别名称
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 构建颜色到类别索引的映射表
colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def voc_label_indices(colormap, colormap2label):
    """将RGB格式的标签图像转换为类别索引矩阵"""
    colormap = np.array(colormap.convert('RGB')).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


# 测试标签转换
y = voc_label_indices(train_labels[0], colormap2label)
print(y[105:115, 130:140], VOC_CLASSES[1])


def voc_rand_crop(feature, label, height, width):
    """同时对原始图像和标签进行同步随机裁剪"""
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        feature, output_size=(height, width)
    )
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label


# 测试随机裁剪并可视化
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 200)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
d2l.plt.show()


class VOCSDataset(torch.utils.data.Dataset):
    """自定义PASCAL VOC2012语义分割数据集类"""

    def __init__(self, is_train, crop_size, voc_dir,
                 colormap2label, max_num=None):
        # ImageNet数据集的RGB通道均值和标准差
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])

        # 图像预处理变换
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])

        self.crop_size = crop_size
        features, labels = read_voc_images(root=voc_dir,
                                           is_train=is_train,
                                           max_num=max_num)
        self.features = self.filter(features)
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label

        print('读取' + str(len(self.features)) + '个有效实例')

    def filter(self, imgs):
        """过滤尺寸小于裁剪大小的图像"""
        return [img for img in imgs if (
                img.size[1] >= self.crop_size[0] and
                img.size[0] >= self.crop_size[1]
        )]

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        feature, label = voc_rand_crop(
            self.features[idx], self.labels[idx], *self.crop_size
        )
        return (self.tsf(feature), voc_label_indices(label, self.colormap2label))

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.features)


# 数据集参数
crop_size = (320, 480)
max_num = 100

# 创建数据集实例
voc_train = VOCSDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = VOCSDataset(False, crop_size, voc_dir, colormap2label, max_num)

# DataLoader参数
batch_size = 64
num_works = 0 if sys.platform.startswith('win32') else 4

# 创建数据迭代器
train_iter = torch.utils.data.DataLoader(
    voc_train, batch_size, shuffle=True,
    drop_last=True, num_workers=num_works
)
test_iter = torch.utils.data.DataLoader(
    voc_test, batch_size, drop_last=True,
    num_workers=num_works
)

# 测试数据迭代器
for X, Y in train_iter:
    print('图像张量：数据类型', X.dtype, '，形状', X.shape)
    print('标签张量：数据类型', Y.dtype, '，形状', Y.shape)
    break
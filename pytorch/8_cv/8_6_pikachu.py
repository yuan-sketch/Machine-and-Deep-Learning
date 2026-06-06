# 袁超
# 开发时间：2026/5/28 20:45
import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import d2lzh8 as d2l
import sys
data_dir = 'data/pikachu'


# 对皮卡丘数据进行读取，并保存为指定格式
class PikachuDetDataset(torch.utils.data.Dataset):
    """
        皮卡丘目标检测数据集类
        参数:
            data_dir: 数据集根目录路径
            part: 数据集划分，只能是'train'(训练集)或'val'(验证集)
            image_size: 输出图片的尺寸，格式为(高度, 宽度)
    """
    def __init__(self, data_dir, part, image_size=(256,256)):
        assert part in ['train', 'val']
        # 图片尺寸
        self.image_size = image_size
        # 图片路径
        self.image_dir = os.path.join(data_dir, part, 'images')
        # 图片标签
        # 读取标签文件：data_dir/part/label.json
        # 标签文件格式：{"1.png": {"class": 0, "loc": [x1, y1, x2, y2]}, ...}
        # 其中loc是归一化后的边界框坐标(0~1之间)
        with open(os.path.join(data_dir, part, 'label.json')) as f:
            self.label = json.load(f)
        self.transform = torchvision.transforms.Compose(
            [
                # ToTensor的作用：
                # 1. 将PIL图片(范围0-255, 形状H×W×C)转换为FloatTensor
                # 2. 像素值归一化到[0.0, 1.0]区间
                # 3. 通道顺序调整为C×H×W(PyTorch默认格式)
                # 将PIL图片转换成围殴于[0.0，1.0]的floattensor,shape(C*H*W)
                torchvision.transforms.ToTensor()
            ]
        )
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        # 构建图片文件名：数据集按1.png, 2.png...顺序命名
        image_path = str(index+1)+'.png'
        # 获取该图片的类别标签(皮卡丘类别为0)
        cls = self.label[image_path]['class']

        # 构建标签数组：[类别, x1, y1, x2, y2]
        # [None, :]将形状从(5,)变为(1, 5)，为后续批量处理做准备
        # dtype设为float32是为了和PyTorch模型输入类型匹配
        label = np.array(
            [cls]+self.label[image_path]['loc'],
            dtype='float32'
        )[None,:]
        # 读取图片并预处理
        # 1. 打开图片文件
        # 2. 转换为RGB格式(确保即使是灰度图也有3个通道)
        # 3. 调整到指定尺寸
        PIL_img = Image.open(
            os.path.join(self.image_dir, image_path)
        ).convert('RGB').resize(self.image_size)
        # 应用定义好的转换，将PIL图片转为PyTorch Tensor
        img = self.transform(PIL_img)
        sample = {
            'label':label,
            'image':img
        }
        return sample
# 随机读取训练集，按序读取测试集
def load_data_pikachu(batch_size, edge_size=256,
                          data_dir='data/pikachu'):
    """
        加载皮卡丘目标检测数据集
        参数:
            batch_size: 每个批次的样本数量
            edge_size: 图片的边长(正方形)
            data_dir: 数据集根目录
        返回:
            train_iter: 训练集数据迭代器
            val_iter: 验证集数据迭代器
    """
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(
            data_dir, 'train', image_size
        )
    val_dataset = PikachuDetDataset(
            data_dir, 'val', image_size
        )
    # 创建训练集DataLoader
    # shuffle=True: 训练时随机打乱样本顺序，防止模型学习到顺序相关的特征
    # num_workers=4: 使用4个子进程加载数据，提高加载速度
    train_iter = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4
        )
    # 创建验证集DataLoader
    # shuffle=False: 验证时不需要打乱样本，按顺序评估即可
    val_iter = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4
        )
    return train_iter, val_iter

if __name__ == '__main__':
    batch_size, edge_size = 32,256
    train_iter, _ = load_data_pikachu(
    batch_size,edge_size,data_dir
    )
    batch = next(iter(train_iter))
    print(batch['image'].shape, batch['label'].shape)
    # 可视化前10张图片及其边界框
    # permute(0,2,3,1)将tensor从(B, C, H, W)转换为(B, H, W, C)
    # 这是因为matplotlib显示图片需要H×W×C的格式
    imgs = batch['image'][:10].permute(0,2,3,1)
    # 获取框坐标
    # 获取前10个样本的边界框坐标
    # 索引说明: [:10]取前10个样本, [0]去掉多余的维度, [1:]取坐标部分(跳过类别)
    bboxes = batch['label'][:10,0,1:]
    axes = d2l.show_images(imgs, 2, 5).flatten()
    for ax, bb in zip(axes, bboxes):
        # 乘以edge_size将归一化坐标(0~1)转换为像素坐标(0~256)
        d2l.show_bboxes(ax, [bb*edge_size], colors=['w'])
    d2l.plt.show()
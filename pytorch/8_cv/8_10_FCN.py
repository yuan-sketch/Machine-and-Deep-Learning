# 袁超
# 开发时间：2026/6/2 12:12
import sys

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import d2lzh8 as d2l

voc_dir = 'data/VOCdevkit/VOC2012'
colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
X = torch.arange(1, 17).view((1,1,4,4)).float()
K = torch.arange(1, 10).view((1,1,3,3)).float()
conv = nn.Conv2d(in_channels=1, out_channels=1,
                 kernel_size=3, bias=False)
conv.load_state_dict({'weight':K})

W, k = torch.zeros((4,16)), torch.zeros(11)
k[:3], k[4:7], k[8:] = K[0,0,0,:], K[0,0,1,:], K[0,0,2,:]
W[0,0:11], W[1,1:12], W[2,4:15], W[3,5:16] = k,k,k,k
print(torch.mm(W, X.view((16,1))).view((1,1,2,2)),'\n',W)

# 构建一个卷积层，输入X的形状为(1,3,64,64),输出Y的通道数增加到10，宽高缩小一半
X = torch.zeros((1,3,64,64))
conv = torch.nn.Conv2d(in_channels=3, out_channels=10,
                       kernel_size=4, padding=1, stride=2)
Y = conv(X)
print(Y.shape)

conv_trans = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3,
                                      kernel_size=4, padding=1, stride=2)
print(conv_trans(Y).shape)


# 构造模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 构造全卷积层net,复制了pretrained_net实例的成员变量features里出去最后两层的所有层以及预训练得到的模型参数
net = nn.Sequential(*list(pretrained_net.children())[:-2])
# 给定输入的高和宽为320和480，net的前向计算将输入的宽和高减小至原来的1/32
X = torch.rand(size=(1,3,320,480))
num_classes = 21
# 在原来网络的基础上添加层
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(
    num_classes, num_classes, kernel_size=64, padding=16,stride=32
))

# 初始化转置卷积层
# 上采样方法——双线性插值
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size+1)//2
    if kernel_size % 2 ==1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1,1),
          torch.arange(kernel_size).reshape(1,-1))
    filt = (1-torch.abs(og[0] - center) / factor) * (1-torch.abs(og[1] - center) / center)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), : , : ] = filt.float()
    return weight

# 构造一个将输入的高和宽放大2倍的转置卷积层，并将其卷积核用bilinear_kernel函数初始化
conv_trans = nn.ConvTranspose2d(3,3,kernel_size=4,padding=1,stride=2,bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3,3,4))
# 读取图像X,上采样结果为Y,调整通道维位置打印图像
img = torchvision.transforms.ToTensor()(d2l.Image.open('data/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1,2,0).detach()

d2l.set_figsize()
print('输入图像尺寸：',img.permute(1,2,0).shape)
d2l.plt.imshow(img.permute(1,2,0))
d2l.plt.show()
print('输出图像尺寸：',out_img.shape)
d2l.plt.imshow(out_img)
d2l.plt.show()


# 全卷积网络中，将转置卷积层初始化为双线性插值的上采样，对于1*1卷积层，采用Xavier随机初始化
W= bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
torch.nn.init.xavier_uniform_(net.final_conv.weight.data);

# 读取数据集
crop_size = (320,480)
batch_size = 32
voc_train = d2l.VOCSegDataset(True, crop_size,voc_dir, colormap2label)
voc_test = d2l.VOCSegDataset(False, crop_size,voc_dir, colormap2label)
num_works = 0 if sys.platform.startswith('win32') else 4
train_iter = torch.utils.data.DataLoader(
    voc_train, batch_size, shuffle=True,drop_last=True, num_workers=num_works
)
test_iter = torch.utils.data.DataLoader(
    voc_test, batch_size, shuffle=False, drop_last=True, num_workers=num_works
)


# 训练模型
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def loss(inputs, targets):
    loss_calc = F.cross_entropy(
        inputs, targets.long(), reduction='none'
    ).mean()
    return loss_calc
num_epochs, lr, wd, devices = 10, 0.003, 1e-3, device
trainer = torch.optim.SGD(net.parameters(), lr, wd)
d2l.train_ch8(train_iter, test_iter, net, loss, trainer, device, num_epochs)

# 预测像素类别
# 将输入图像在各个通道做标准化，并转成四维输入格式
def predict(img):
    rgb_mean = np.array([0.485,0.456,0.406])
    rgb_std = np.array([0.229,0.224,0.225])
    tsf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=rgb_mean,std=rgb_std)
        ]
    )
    X = tsf(img).unsqueeze(0)
    pred = net(X.to(device)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X,:]

# 从图像左上角开始裁剪形状为320*480的区域：只有该区域用于预测
test_images, test_labels = d2l.read_voc_images(is_train=False)
n ,imgs = 5, []
for i in range(n):
    crop_rect = (0,0,320,480)
    X = torchvision.transforms.functional.crop(
        test_images[i], *crop_rect
    )
    pred = label2image(predict(X))
    imgs += [torch.from_numpy(np.array(X)),
             pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect
             )]
d2l.show_images(imgs[::3]+imgs[1::3]+imgs[2::3], 3, n, scale=2)
d2l.plt.show()


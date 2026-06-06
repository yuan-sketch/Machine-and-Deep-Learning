# 袁超
# 开发时间：2026/5/9 14:19
import sys
import time
import torch
import torchvision.datasets
from torch import nn,optim
from torchvision import transforms
from my_utils_CNN import train_ch5
import torch
import sys

print("1. Python解释器路径:", sys.executable)
print("2. PyTorch完整版本:", torch.__version__)
print("3. CUDA是否可用:", torch.cuda.is_available())
print("4. PyTorch编译用CUDA版本:", torch.version.cuda)
print("5. 检测到的GPU数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i} 型号: {torch.cuda.get_device_name(i)}")

    print("=" * 60)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"最终使用的训练设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print("=" * 60)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,96,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，
            # 进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合

        self.fc = nn.Sequential(
            nn.Linear(256*5*5,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )
    def forward(self,imag):
        feature = self.conv(imag)
        output = self.fc(feature.view(imag.shape[0],-1))
        return output


net = AlexNet()
print(net)

def load_data_fashion_mnist(batch_size, resize=None):
    """
    function：
    将fashion mnist数据集划分为小批量样本
    Parameters:
    batch_size - 小批量样本的大小(int)
    resize - 对图像的维度进行扩大
    Returns:
    train_iter - 训练集样本划分为最小批的结果
    test_iter - 测试集样本划分为最小批的结果
    Modify:添加图像维度变化
    """
    # 存储图像处理流程
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='D:\桌面\吴恩达机器学习笔记等资源\code\pytorch\data/FashionMNIST',
        train=True,
        download=False,
        transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='D:\桌面\吴恩达机器学习笔记等资源\code\pytorch\data/FashionMNIST',
        train=False,
        download=False,
        transform=transform
    )
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_iter,test_iter


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size,224)


#训练模型
lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(),lr)
train_ch5(net, train_iter,test_iter,batch_size,optimizer,device,num_epochs)

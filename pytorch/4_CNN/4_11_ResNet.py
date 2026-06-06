# 袁超
# 开发时间：2026/5/12 14:00
import torch
import torch.nn.functional as F
from torch import nn
from my_utils_CNN import GlobalAvgPool2d,FlattenLayer,load_data_fashion_mnist,train_ch5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Residual(nn.Module):
    #输入通道数，输出通道数，是否使用1*1卷积核，步长
    def __init__(self, in_channels, out_channels, use_1x1conv = False, stride = 1):
        super(Residual,self).__init__()
        #3x3搭配1步长，特征图大小不变
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1,stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)

'''blk = Residual(3, 3)
X = torch.rand((4,3,6,6))
print(blk(X).shape)
blk = Residual(3, 6, use_1x1conv=True, stride=2)
print(blk(X).shape)'''

net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

def resnet_block(in_channels, out_channels,
                 num_residuals, first_block = False):
    if first_block:
        #第一个模块的通道数同输入通道数一致
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels,
                                use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net.add_module('resnet_block1', resnet_block(64,64,2,first_block=True))
net.add_module('resnet_block2', resnet_block(64,128,2))
net.add_module('resnet_block3', resnet_block(128,256,2))
net.add_module('resnet_block4',resnet_block(256,512,2))

net.add_module('global_avg_pool',GlobalAvgPool2d())
net.add_module('fc',nn.Sequential(FlattenLayer(),nn.Linear(512,10)))

X = torch.rand((1,1,224,224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, 'output shape:\t',X.shape)


def resnet18(output=10, in_channels = 3):
    """
    function：
    18层残差网络
    Parameters:
    in_channels - 输入通道数
    out_channels - 输出通道数
    Returns:
    残差网络
    """
    net = nn.Sequential(
        nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3
        ),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module('resnet_block1',resnet_block(64,64,2,first_block=True))
    net.add_module('resnet_block2', resnet_block(64, 128, 2, first_block=True))
    net.add_module('resnet_block3', resnet_block(128, 256, 2, first_block=True))
    net.add_module('resnet_block4', resnet_block(256, 512, 2, first_block=True))

    net.add_module('global_avg_pool',GlobalAvgPool2d())
    net.add_module('fc',nn.Sequential(FlattenLayer(),nn.Linear(512,output)))

    return net


#训练模型
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
lr, num_epochs = 0.001,5
optimizer = torch.optim.Adam(net.parameters(),lr)
train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)


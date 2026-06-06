# 袁超
# 开发时间：2026/5/12 15:09
import torch
from torch import nn
from my_utils_CNN import GlobalAvgPool2d,FlattenLayer,load_data_fashion_mnist,train_ch5

#稠密层
#批量统一化，激活，卷积
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, padding=1)
    )
    return blk

#在前向计算时，我们将每块的输入和输出在通道维上联结
class DenseBlock(nn.Module):
    def __init__(self,num_convs, in_channels, out_channels):
        super(DenseBlock,self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i*out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        #计算输出通道
        self.out_channels = in_channels + num_convs * out_channels
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            #在通道维上将输入和输出连结
            X = torch.cat((X,Y),dim=1)

        return X

blk = DenseBlock(2,3,10)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

#过渡层
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk

#DenseNet模型
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

#Num_channels为当前通道数
num_channels, growth_rate = 64,32
num_convs_in_dense_blocks = [4,4,4,4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module('DenseBlock_%d'%i,DB)
    #上一个稠密块的输出通道数
    num_channels = DB.out_channels
    #在稠密块之间加入通道数减半的过渡层
    if i !=len(num_convs_in_dense_blocks)-1:
        net.add_module('transition_block_%d'%i,
                       transition_block(num_channels, num_channels//2))
        num_channels = num_channels//2
net.add_module('BN',nn.BatchNorm2d(num_channels))
net.add_module('relu',nn.ReLU())
#GlobalAvgPool2d的输出：（Batch,num_channels,1,1）
net.add_module('global_avg_pool',GlobalAvgPool2d())
net.add_module('fc',nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_channels,10)
))


#训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
lr, num_epochs = 0.001,5
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
train_ch5(net, train_iter,test_iter,batch_size,optimizer,device,num_epochs)
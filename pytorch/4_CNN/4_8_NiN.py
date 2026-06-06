# 袁超
# 开发时间：2026/5/9 17:05
import torch
from torch import nn
import torch.nn.functional as F
from my_utils_CNN import FlattenLayer,load_data_fashion_mnist,train_ch5

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
def nin_block(in_channels, out_channels, kernal_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, stride=stride,padding=padding),
        nn.ReLU(),
        #相当于全连接层
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU()
    )
    return blk

class GlobalAvgPool2d(nn.Module):
    #全局平均池化将窗口形设置成高和宽
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

net = nn.Sequential(
    nin_block(1, 96, kernal_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96,256,kernal_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(256,384,kernal_size=3,stride=1,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Dropout(0.5),
    nin_block(384,10,kernal_size=3,stride=1,padding=1),
    GlobalAvgPool2d(),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    FlattenLayer()
)

X = torch.rand(1,1,224,224)
for name,blk in net.named_children():
    X = blk(X)
    print(name,'out sahpe:',X.shape)



#训练模型
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size,224)
lr ,num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(),lr)
train_ch5(net, train_iter,test_iter,batch_size,optimizer,device, num_epochs)
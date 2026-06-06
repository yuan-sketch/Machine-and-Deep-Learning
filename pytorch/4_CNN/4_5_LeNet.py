# 袁超
# 开发时间：2026/5/9 11:59
import time

import torch
from torch import nn,optim
from my_utils_CNN import load_data_fashion_mnist


print("="*50)
import torch
import sys

print("1. Python路径:", sys.executable)
print("2. PyTorch版本:", torch.__version__)
print("3. CUDA是否可用:", torch.cuda.is_available())
print("4. CUDA版本(编译用):", torch.version.cuda)
try:
    print("5. GPU驱动版本:", torch.cuda.get_driver_version())
except Exception as e:
    print("5. 无法获取驱动版本:", e)
print("="*50)


device = torch.device('cuda'if torch.cuda.is_available()else'cpu')
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            #in_channels,out_channels,kernal_size
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            #kernal_size,stride
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )

    def forward(self,imag):
        feature = self.conv(imag)
        output = self.fc(feature.view(imag.shape[0], -1))
        return output

net = LeNet()
print(net)

#实验LeNet模型
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

#模型较复杂，利用GPU计算
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        #如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    for X,y in data_iter:
        if isinstance(net, torch.nn.Module):
            #评估模式，这会关闭dropout
            net.eval()
            #.cpu()保证可以进行数值加减
            acc_sum +=(net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            #改回训练模式
            net.train()
        else:
            if('is_training' in net.__code__.co_varnames):
                acc_sum +=(net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n +=y.shape[0]
    return acc_sum/n

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    """
    function：
    利用softmax回归模型对图像进行分类识别
    学习率采用0.001，训练算法使用Adam算法，损失函数使用交叉熵损失函数。
    小结：
    Parameters:
    net - 定义的网络
    train_iter - 训练集样本划分为最小批的结果
    test_iter - 测试集样本划分为最小批的结果
    num_epochs - 迭代次数
    batch_size - 最小批大小
    optimizer - 优化器
    device - 指定计算在GPU或者CPU上进行
    Returns:
    """
    net = net.to(device)
    print("training on",device)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count = 0.0,0.0,0,0
        start = time.time()
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            #梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum +=(y_hat.argmax(dim=1) == y).sum().cpu().item()
            n +=y.shape[0]
            batch_count +=1
        test_acc = evaluate_accuracy(test_iter, net,device=device)
        print('epoch %d, loss %f, train acc %.3f,test acc %.3f,\time %.1f sec'
              %(epoch+1, train_l_sum/batch_count, train_acc_sum/n,  test_acc, time.time()-start ))


lr, num_epochs = 0.001,5
optimizer = optim.Adam(net.parameters(), lr)
train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)
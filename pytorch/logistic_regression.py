import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import d2l
#1. 加载数据集--并查看数据情况
mnist_train = torchvision.datasets.FashionMNIST(
    root='data/FashionMNIST', train=True, download=True,
    transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(
    root='data/FashionMNIST', train=False, download=True,
    transform=transforms.ToTensor())

print(len(mnist_train), len(mnist_test))
feature, label = mnist_train[0]
print(feature.shape, feature.dtype)
print(label, type(label))

# ----------------------
# 画图函数（纯 plt，彻底不用 d2l）
# ----------------------
def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy(), cmap='gray')
        f.set_title(lbl)
        f.axis('off')
    plt.show()

# ----------------------
# 标签（完整 10 类）
# ----------------------
def get_fashion_mnist_labels(labels):
    text_labels = ['t_shirt', 'trouser', 'pullover','dress', 'coat', 'sandal',
                   'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in labels]

# ----------------------
# 取前 10 张图
# ----------------------
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])

# 画图
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# ----------------------
# 数据加载速度测试
# ----------------------
batch_size = 256
train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))



import torch
from torch import nn
from torch.nn import init

def load_data_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="data/FashionMNIST", train=True, transform=transform, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="data/FashionMNIST", train=False, transform=transform, download=False)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter

# 调用（和 d2l 一模一样！）
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

#x的形状由（batch_size,1,28,28)转换成（batch_size,784)
class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def forward(self,x):
        y = self.linear(x.view(x.shape[0],-1))
        return y
net = LinearNet(num_inputs,num_outputs)

#将对 x 的形状转换的这个功能自定义一个 FlattenLayer 并记录在 d2l 中方便后面使用。
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)
from collections import OrderedDict
net = nn.Sequential(
    OrderedDict(
        [
            ('flatten',FlattenLayer()),
            ('linear',nn.Linear(num_inputs,num_outputs))
        ]
    )
)
#使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数。
init.normal_(net.linear.weight,mean=0,std=0.01)
init.constant_(net.linear.bias,val=0)
#交叉熵损失函数
loss = nn.CrossEntropyLoss()
#定义优化算法
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
#训练模型
num_epochs = 5
def train_ch3(net,train_iter, test_iter, loss, num_epochs, batch_size,
              params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()

            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            optimizer.step()

            train_l_sum +=l.item()
            train_acc_sum +=(y_hat.argmax(dim=1) == y).sum().item()
            n +=y.shape[0]

        test_acc = evaluate_accuracy(net,test_iter)

        print(f'epoch {epoch+1},loss {train_l_sum/n:.4f},train acc {train_acc_sum/n:.3f},'
              f'test acc {test_acc:.3f}')
def evaluate_accuracy(net,data_iter):
    acc_sum,n = 0.0,0
    net.eval()
    with torch.no_grad():
        for X,y in data_iter:
            acc_sum +=(net(X).argmax(dim=1) == y).float().sum().item()
            n +=y.shape[0]
    return acc_sum/n

train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)

#预测

X,y = next(iter(test_iter))
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true +'\n' + pred for true ,pred in
          zip(true_labels, pred_labels)]
show_fashion_mnist(X[:9],titles[:9])
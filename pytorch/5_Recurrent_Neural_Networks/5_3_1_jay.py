# 袁超
# 开发时间：2026/5/17 22:13
import torch
import numpy as np
from my_utils_RNN import load_data_jay_lyrics,sgd


#读取数据集
import zipfile
with zipfile.ZipFile('data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
#print(corpus_chars[:40])
corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')
corpus_chars = corpus_chars[:10000]
#建立字符索引
#每个字符映射成一个正整数，构成字典
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i ,char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(vocab_size)

corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:',''.join([idx_to_char[idx] for idx in sample]))
print('indices:',sample)

#时序数据的采样
#1.随机采样
import random
def data_iter_random(corpus_indices, batch_size,
                     num_steps, device=None):
    #输出的索引x是输入索引y加1
    #计算一共生成多少样本
    num_examples = (len(corpus_indices) - 1) // num_steps
    #循环多少次可以遍历所有样本
    epoch_size = num_examples // batch_size
    #每一条样本的索引
    example_indices = list(range(num_examples))
    #随机打乱样本索引
    random.shuffle(example_indices)
    #返回从pos开始到pos+num_steps-1的序列
    def _data(pos):
        return corpus_indices[pos : pos+num_steps]
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    #构建每一个epoch内的样本
    for i in range(epoch_size):
        #每次读取batch_size个随机样本
        i = i*batch_size
        batch_indices = example_indices[i : i+batch_size]
        X = [_data(j*num_steps) for j in batch_indices]
        y = [_data(j*num_steps+1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device),torch.tensor(y, dtype=torch.float32, device=device)

'''my_seq = list(range(30))
for X,y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X:', X ,'\ny:', y , '\n')
'''
#2.相邻采样
def data_iter_consecutive(corpus_indices, batch_size,
                          num_steps, device=None):
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    #原始数据长度
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0 : batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i*num_steps
        X = indices[: , i : i + num_steps]
        y = indices[: , i+1 : i+num_steps+1]
        yield X,y

'''for X, y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X:', X , '\ny:', y , '\n')
'''

#从零开始实现RNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()


import torch.nn.functional as F
print(F.one_hot(torch.tensor([0,2,1]), vocab_size))          #建立一个长为vocab_size的全0向量，索引位置是0和2

def to_onehot(X, size):
    return F.one_hot(X.t(), size)
X = torch.arange(10).view(2,5)
inputs = to_onehot(X, vocab_size)
print(len(inputs), inputs)


#初始化模型参数
num_inputs , num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use:', device)
def get_params():
    def _one(shape):
        ts = torch.tensor(
            np.random.normal(0, 0.01, size=shape),
            device = device,
            dtype= torch.float32
        )
        return torch.nn.Parameter(ts, requires_grad=True)
    #隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(
        torch.zeros(num_hiddens ,device=device, requires_grad=True)
    )
    #输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(
        torch.zeros(num_outputs, device=device,requires_grad=True)
    )
    return torch.nn.ParameterList([W_xh,W_hh,b_h,W_hq,b_q])


#定义模型
#定义函数返回初始化的隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

#在一个时间步内计算隐藏状态和输出
def rnn(inputs, state, params):
    # inputs和outputs是num_steps个形状为(batch_size,vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    #上一层传来的隐藏状态
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(
            torch.matmul(X.float(),W_xh) +
            torch.matmul(H, W_hh) + b_h
        )
        Y = torch.matmul(H,W_hq) + b_q
        outputs.append(Y)
    return outputs,(H, )

state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
#print('\n\n',outputs,'\n\n', outputs[0].shape, state_new[0].shape)


#定义预测函数
def predict_rnn(prefix, num_chars, rnn, params,
                init_rnn_state, num_hiddens, vocab_size,
                device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    # 将输入的首字符传入到输出序列中
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) -1):
        # 计算上一时间步的输出作为当前时间步的输入
        X = to_onehot(
            torch.tensor([[output[-1]]], device=device),
            vocab_size
        )
        # 计算输出并更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))

    return ''.join([idx_to_char[i] for i in output])

print(predict_rnn('分开',10,rnn,params,init_rnn_state,
                  num_hiddens, vocab_size,device,
                  idx_to_char,char_to_idx))

#裁剪梯度
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm +=(param.grad.data**2).sum()
    norm = norm.sqrt().item()
    if norm >theta:
        for param in params:
            param.grad.data *=(theta/norm)
    return param


#定义模型训练函数
import time
import math
# rnn——要使用的rnn模型
# get_params——初始化模型参数
# init_rnn_state——初始化隐藏状态
# num_hiddens——隐藏神经元个数
# vocab_size——字典大小
# device——指定计算在GPU或CPU上
# corpus_indices——编码之后的样本
# idx_to_char——索引到字符之间的映射
# char_to_idx——字符到索引之间的映射
# is_random_iter——是否采用随机采样
# num_epochs——训练轮次
# num_step——每一批中每个样本的大小
# lr——学习率
# chipping_theta——梯度裁剪阈值
# batch_size——批量大小
# pred_period——每多少次打印训练结果
# pred_len——生成样本长度
# prefixes——传入的生成引导内容
def train_and_predict_rnn(rnn, get_params, init_rnn_state,
                          num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes):
    if is_random_iter:
        # 随机采样
        data_iter_fn = data_iter_random
    else:
        # 相邻采样
        data_iter_fn = data_iter_consecutive
    #初始化模型参数
    params = get_params()
    #交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(
                batch_size, num_hiddens, device)
        #损失之和，样本数，起始时间
        l_sum, n, start = 0.0,0,time.time()
        # 生成训练样本及label
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                for s in state:
                    s.detach()
            #独热编码
            inputs = to_onehot(X.long(), vocab_size)
            # outputs有num_steps个形状为(batch_size,vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps*batch_size,vocab_size)
            outputs = torch.cat(outputs,dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为batch*num_steps的向量，这样跟输出行一一对应
            y = torch.transpose(Y, 0,1).contiguous().view(-1)
            l = loss(outputs, y.long())
            #梯度清零
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward(retain_graph = True)
            # 裁剪梯度
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr,1)
            l_sum +=l.item()*y.shape[0]
            n +=y.shape[0]

        if(epoch+1)%pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec'%
                  (epoch+1,math.exp(l_sum/n),time.time()-start))
            for prefix in prefixes:
                print(" -", predict_rnn(prefix, pred_len, rnn, params,
                                        init_rnn_state, num_hiddens,
                                        vocab_size, device, idx_to_char,char_to_idx))


num_epochs, num_steps, batch_size, lr = 250,35,32,1e2
clipping_theta = 1e-2
pred_period, pred_len, prefixes = 50,50,['分开','不分开']

train_and_predict_rnn(rnn, get_params, init_rnn_state,
                          num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx, False,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)



# 袁超
# 开发时间：2026/5/17 23:06
import torch
import torch.nn.functional as F
import time
import math
from torch import nn


def load_data_jay_lyrics():
    import zipfile
    with zipfile.ZipFile('data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    # print(corpus_chars[:40])
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:10000]
    # 建立字符索引
    # 每个字符映射成一个正整数，构成字典
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    #print(vocab_size)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    #print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    #print('indices:', sample)
    return corpus_indices,char_to_idx,idx_to_char,vocab_size

def to_onehot(X, size):
    return F.one_hot(X.t(), size)


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

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


#定义预测函数
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
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


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel,self).__init__()
        self.rnn = rnn_layer
        # 若为双向循环网络需要x2
        self.hidden_size = rnn_layer.hidden_size*(
            2 if rnn_layer.bidirectional else 1
        )
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    # inputs:(batch, state)
    def forward(self, inputs, state):
        X = to_onehot(inputs.long(), self.vocab_size)
        Y, self.state = self.rnn(X.float(), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size,
                        device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = torch.tensor([output[-1]], device=device).view(1,1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y,state) = model(X, state)
        if t<len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model,num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes):
    # 相邻采样
    data_iter_fn = data_iter_consecutive
    #交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        #损失之和，样本数，起始时间
        l_sum, n, start = 0.0,0,time.time()
        # 生成训练样本及label
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            # outputs为(batch_size*num_steps,vocab_size)的矩阵
            (outputs, state) = model(X, state)
            # 转置后再变成长度为batch*num_steps的向量，这样跟输出行一一对应
            y = torch.transpose(Y, 0,1).contiguous().view(-1)
            l = loss(outputs, y.long())
            #梯度清零
            optimizer.zero_grad()
            l.backward()
            # 裁剪梯度
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum +=l.item()*y.shape[0]
            n +=y.shape[0]
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if(epoch+1)%pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec'%
                  (epoch+1,perplexity,time.time()-start))
            for prefix in prefixes:
                print(" -", predict_rnn_pytorch(prefix, pred_len, model,
                                        vocab_size, device, idx_to_char,char_to_idx))

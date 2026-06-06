# 袁超
# 开发时间：2026/5/18 20:50
import torch
from my_utils_RNN import load_data_jay_lyrics,to_onehot,grad_clipping,sgd,data_iter_consecutive
import torch.nn.functional as F
from torch import nn
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

#使用Tensor初始化隐藏状态,(隐藏层个数，批量大小，隐藏单元个数)
state = torch.zeros((1,2,num_hiddens))

num_steps = 32
batch_size = 2
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape,'\n',len(state_new),'\n',state_new[0].shape)


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

model = RNNModel(rnn_layer, vocab_size).to(device)
print(predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

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

num_epochs, num_steps, batch_size, lr = 250,35,32,1e-2
clipping_theta = 1e-2
pred_period, pred_len, prefixes = 50,50,['分开','不分开']

train_and_predict_rnn_pytorch(model,num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)

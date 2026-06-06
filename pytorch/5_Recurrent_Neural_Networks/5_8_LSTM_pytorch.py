# 袁超
# 开发时间：2026/5/19 16:55
import torch
from torch import nn
import numpy as np
from my_utils_RNN import load_data_jay_lyrics,train_and_predict_rnn_pytorch,RNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 读取数据集
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

num_epochs, num_steps, batch_size, lr = 200, 35, 32, 1e-2
clipping_theta = 1e-2
pred_period, pred_len, prefixes = 40,50,['分开','不分开']


lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(lstm_layer, vocab_size)
train_and_predict_rnn_pytorch(
                          model,num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes
)

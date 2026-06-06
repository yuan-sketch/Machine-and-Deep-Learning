# 袁超
# 开发时间：2026/5/19 15:08
import torch
from my_utils_RNN import RNNModel,train_and_predict_rnn_pytorch,load_data_jay_lyrics
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(gru_layer, vocab_size).to(device)
num_epochs, num_steps, batch_size, lr = 250, 35, 32, 1e-2
clipping_theta = 1e-2
pred_period, pred_len, prefixes = 40,50,['分开','不分开']
train_and_predict_rnn_pytorch(model,num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)


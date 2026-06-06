# 袁超
# 开发时间：2026/5/19 13:39
import torch
import numpy as np
from my_utils_RNN import load_data_jay_lyrics,train_and_predict_rnn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

#初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use',device)
def get_params():
    def _one(shape):
        ts = torch.tensor(
            np.random.normal(0, 0.01,size=shape),
            device=device,dtype=torch.float32
        )
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens,num_hiddens)),
                torch.nn.Parameter(
                    torch.zeros(
                        num_hiddens,
                        device=device,
                        dtype=torch.float32
                    ),
                    requires_grad=True
                ))

    #更新门参数
    W_xz, W_hz, b_z = _three()
    #重置门参数
    W_xr, W_hr, b_r = _three()
    #候选隐藏状态参数
    W_xh, W_hh, b_h = _three()
    #输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(
        torch.zeros(num_outputs, device=device, dtype=torch.float32),
        requires_grad=True
    )
    return torch.nn.ParameterList(
        [
            W_xz, W_hz, b_z,
            W_xr, W_hr, b_r,
            W_xh, W_hh, b_h,
            W_hq, b_q
        ]
    )

#定义模型
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens),device=device), )

def gru(inputs, state, params):
    #参数
    W_xz, W_hz, b_z,W_xr, W_hr, b_r,W_xh , W_hh,b_h ,W_hq, b_q = params
    #初始化的隐藏状态
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(
            torch.matmul(X.float(),W_xz)+torch.matmul(H, W_hz)+b_z
        )
        R = torch.sigmoid(
            torch.matmul(X.float(), W_xr)+torch.matmul(H, W_hr)+b_r
        )
        H_tilda = torch.tanh(
            torch.matmul(X.float(), W_xh)+torch.matmul(R*H, W_hh)+b_h
        )
        H = Z*H +(1-Z)*H_tilda
        Y = torch.matmul(H, W_hq)+b_q
        outputs.append(Y)
    return outputs, (H.detach(),)

#训练模型
num_epochs, num_steps, batch_size, lr = 250, 35, 32, 1e2
clipping_theta = 1e-2
pred_period, pred_len, prefixes = 40,50,['分开','不分开']

train_and_predict_rnn(gru, get_params, init_gru_state,
                          num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx, False,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)

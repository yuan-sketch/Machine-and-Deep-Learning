# 袁超
# 开发时间：2026/5/21 14:44
import d2lzh6 as d2l
import torch
def f_2d(x1, x2):
    return 0.1 * x1**2 + 2 * x2**2

def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1-v1, x2-v2, v1, v2
eta, gamma = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()

# 动量法需要对每一个自变量维护一个同他一样形状的速度变量
# 超参数多了动量参数
features, labels = d2l.get_data_ch7()
# 初始化动量相关变量
def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1),  dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p,v in zip(params, states):
        v.data = hyperparams['momentum']*v.data+\
            hyperparams['lr']*p.grad.data
        p.data -=v.data

d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr':0.02,'momentum':0.5},
              features, labels)

d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr':0.02,'momentum':0.9},
              features, labels)

d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr':0.004,'momentum':0.9},
              features, labels)
d2l.plt.show()
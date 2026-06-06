# 袁超
# 开发时间：2026/5/21 15:30
import torch
import d2lzh6 as d2l
features, labels = d2l.get_data_ch7()
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr':0.004,'momentum':0.9},
                      features, labels)
d2l.plt.show()
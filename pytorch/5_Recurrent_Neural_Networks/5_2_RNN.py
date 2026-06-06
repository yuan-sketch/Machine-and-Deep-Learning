# 袁超
# 开发时间：2026/5/17 20:14
import torch
X, W_xh = torch.rand(3,1), torch.rand(1,4)
H, W_hh = torch.rand(3,4), torch.rand(4,4)
y1 = torch.matmul(X,W_xh)+torch.matmul(H,W_hh)
print(y1)
y2 = torch.matmul(torch.cat((X,H),dim=1),
                  torch.cat((W_xh,W_hh),dim=0))
print(y2)

# 袁超
# 开发时间：2026/5/8 14:00
import torch

print(
    torch.device('cpu'),
    torch.cuda.device('cuda'),
    torch.cuda.device('cuda:1')
)

a = torch.tensor([1,2,3],device=torch.device('cuda'))
print(a)
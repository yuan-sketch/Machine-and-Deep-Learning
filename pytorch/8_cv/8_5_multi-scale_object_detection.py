# 袁超
# 开发时间：2026/5/28 15:14
from PIL import Image
import d2lzh8 as d2l
import torch


img = Image.open('data/catdog.jpg')
w, h = img.size
print(w,h)
def display_anchors(fmap_w, fmap_h, s):
    # 批量，通道数，高，宽
    fmap = torch.zeros((1,10,fmap_h,fmap_w), dtype=torch.float32)
    # 归一化
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h
    # 平移锚框，使其均匀分布在图片上
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1,2,0.5])
    anchors += torch.tensor([offset_x/2, offset_y/2,
                             offset_x/2, offset_y/2])
    bbox_scale = torch.tensor([[w,h,w,h]], dtype=torch.float32)
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0]*bbox_scale)
    d2l.plt.show()

display_anchors(fmap_w=4,fmap_h=2,s=[0.15])
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
display_anchors(fmap_w=1, fmap_h=1,s=[0.8])



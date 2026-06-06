# 袁超
# 开发时间：2026/5/25 12:11
import d2lzh8 as d2l
from PIL import Image

d2l.set_figsize()
img  = Image.open('data/catdog.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

#[左上角x, 左上角y,右下角x,右下角y]
dog_bbox ,cat_bbox = [60,45,378,526],[400,114,655,493]
# 将[左上角x, 左上角y,右下角x,右下角y]格式，转换成((左上x,左上y),宽,高)形式
def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(
        xy = (bbox[0], bbox[1]),width=bbox[2]-bbox[0],
        height=bbox[3]-bbox[1],
        fill=False,
        edgecolor=color, linewidth=2
    )

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox,'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox,'red'))
d2l.plt.show()
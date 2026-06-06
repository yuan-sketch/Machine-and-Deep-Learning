# 袁超
# 开发时间：2026/5/20 15:19
import d2lzh6 as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch


def f(X):
    return X * np.cos(np.pi*X)

d2l.set_figsize((4.5, 2.5))
X = np.arange(-1.0, 2.0, 0.1)
# 只取返回列表中的第一个元素
fig, = d2l.plt.plot(X, f(X))
fig.axes.annotate('loacal mininum', xy=(-0.3, -0.25),
                  xytext=(-0.77, -1.0),
                  arrowprops = dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95),
                  xytext=(0.6, 0.8),
                  arrowprops=dict(arrowstyle='->'))
d2l.plt.xlabel('X')
d2l.plt.ylabel('f(X)')
d2l.plt.show()

X = np.arange(-2.0, 2.0, 0.1)
fig,  = d2l.plt.plot(X, X**3)
# xy:箭头坐标，xytext:文字添加位置
fig.axes.annotate('saddle point', xy=(0, -0.2),
                  xytext=(-0.52, -5.0),
                  arrowprops=dict(arrowstyle='->'))
d2l.plt.xlabel('X')
d2l.plt.ylabel('f(X)')
d2l.plt.show()

# 二维空间鞍点
X, y = np.mgrid[-1:1:31j, -1:1:31j]
z = X**2 - y**2
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(X, y, z, **{'rstride':2, 'cstride': 2})
ax.plot([0], [0], [0],'rx')
ticks = [-1,0,1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('X')
d2l.plt.ylabel('y')
d2l.plt.show()
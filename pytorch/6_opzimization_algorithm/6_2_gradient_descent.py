# 袁超
# 开发时间：2026/5/20 16:24
import math
import d2lzh6 as d2l
import numpy as np


def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -=eta *2 * x
        results.append(x)
    print('epoch 10, x:',x)
    return results
res = gd(0.2)

def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x*x for x in f_line])
    d2l.plt.plot(res, [x*x for x in res], '-o')
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('f(X)')

show_trace(res)
d2l.plt.show()



# 多维梯度下降
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2 ,0 ,0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1,x2,s1,s2)
        results.append((x1,x2))
    print('epoch %d, x1 %f, x2 %f'%(i+1, x1, x2))
    return results
def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(
        np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1)
    )
    d2l.plt.contour(x1, x2, f(x1,x2), colors = '#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    d2l.plt.show()
eta = 0.1

def f_2d(x1,x2):
    return x1**2+2*x2**2

def gd_2d(x1,x2,s1,s2):
    return (x1-2*eta*x1, x2-4*eta*x2,0,0)

show_trace_2d(f_2d, train_2d(gd_2d))



# 随机梯度下降
def sgd_2d(x1,x2,s1,s2):
    return (x1-eta*(2*x1+np.random.normal(0.1)),
            x2-eta*(4*x2+np.random.normal(0.1)),0,0)

show_trace_2d(f_2d, train_2d(sgd_2d))
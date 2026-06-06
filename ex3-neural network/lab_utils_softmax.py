# 袁超
# 开发时间：2026/4/13 16:02
# 吴恩达机器学习/深度学习课程工具函数
# 适用于 Softmax 多分类实验
# 包含：plt_mc, plt_cat_mc, plt_layer_relu, plt_out, relu, softmax

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ------------------------------------------------------------------------------
# 1. 绘制原始多分类数据集
# ------------------------------------------------------------------------------
def plt_mc(X, y, classes, centers, std):
    cmp = ListedColormap(['blue','orange','green','purple'])
    plt.figure(figsize=(6,4))
    for i in range(classes):
        plt.scatter(X[y==i,0], X[y==i,1], color=cmp(i), label=f"class {i}", s=30)
    plt.title("Multiclass Data")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# ------------------------------------------------------------------------------
# 2. 绘制多分类决策边界
# ------------------------------------------------------------------------------
def plt_cat_mc(X, y, model, classes):
    xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.02),
                         np.arange(ymin, ymax, 0.02))
    points = np.c_[xx.ravel(), yy.ravel()]
    logits = model.predict(points, verbose=0)
    yhat = np.argmax(logits, axis=1)

    cmp = ListedColormap(['blue','orange','green','purple'])
    plt.contourf(xx, yy, yhat.reshape(xx.shape), cmap=cmp, alpha=0.3)
    for i in range(classes):
        plt.scatter(X[y==i,0], X[y==i,1], color=cmp(i), label=f"class {i}", s=40)
    plt.title("Decision Boundary (Multiclass)")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.axis('equal')
    plt.show()

# ------------------------------------------------------------------------------
# 3. 绘制第一层 ReLU 层输出（你指定参数：X,y,w1,b1,classes）
# ------------------------------------------------------------------------------
def plt_layer_relu(X, y, w1, b1, classes):
    z1 = X @ w1.T + b1
    a1 = np.maximum(0, z1)   # ReLU

    cmp = ListedColormap(['blue','orange','green','purple'])
    plt.figure(figsize=(6,4))
    for i in range(classes):
        plt.scatter(a1[y==i,0], a1[y==i,1], color=cmp(i), label=f"class {i}", s=40)
    plt.title("Layer 1 Output (ReLU Activated)")
    plt.xlabel("a1_0")
    plt.ylabel("a1_1")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------------------------
# 4. 绘制第二层输出（softmax前的logits）
# ------------------------------------------------------------------------------
def plt_out(X, y, w1, b1, w2, b2, classes):
    z1 = X @ w1.T + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2.T + b2

    cmp = ListedColormap(['blue','orange','green','purple'])
    plt.figure(figsize=(6,4))
    for i in range(classes):
        plt.scatter(z2[y==i,0], z2[y==i,1], color=cmp(i), label=f"class {i}", s=40)
    plt.title("Layer 2 Output (Logits)")
    plt.xlabel("z2_0")
    plt.ylabel("z2_1")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------------------------
# 5. ReLU 激活函数
# ------------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)

# ------------------------------------------------------------------------------
# 6. Softmax 函数
# ------------------------------------------------------------------------------
# 修复完成！可直接运行的 plt_softmax
def plt_softmax(softmax_func):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots(figsize=(7, 4))
    plt.subplots_adjust(bottom=0.3)

    z = np.linspace(-10, 10, 100)
    zv = np.array([0., 0., 0.])

    ax.set_ylim(0, 1.1)
    ax.set_title('Softmax Function')
    ax.set_xlabel('z')
    ax.set_ylabel('softmax(z)')

    line0, = ax.plot(z, np.zeros_like(z), lw=2, color='blue', label='a0')
    line1, = ax.plot(z, np.zeros_like(z), lw=2, color='orange', label='a1')
    line2, = ax.plot(z, np.zeros_like(z), lw=2, color='green', label='a2')
    ax.legend()

    ax_z0 = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_z1 = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_z2 = plt.axes([0.25, 0.05, 0.65, 0.03])

    z0_slider = Slider(ax=ax_z0, label='z0', valmin=-10, valmax=10, valinit=0)
    z1_slider = Slider(ax=ax_z1, label='z1', valmin=-10, valmax=10, valinit=0)
    z2_slider = Slider(ax=ax_z2, label='z2', valmin=-10, valmax=10, valinit=0)

    def update(val):
        zv[0] = z0_slider.val
        zv[1] = z1_slider.val
        zv[2] = z2_slider.val

        a = softmax_func(zv)

        line0.set_ydata([a[0]] * len(z))
        line1.set_ydata([a[1]] * len(z))
        line2.set_ydata([a[2]] * len(z))

        fig.canvas.draw_idle()

    z0_slider.on_changed(update)
    z1_slider.on_changed(update)
    z2_slider.on_changed(update)
    update(None)
    plt.show()

# ------------------------------------------------------------------------------
# 7. 手动完整前向传播（用于核对模型输出）
# ------------------------------------------------------------------------------
def my_forward(X, w1, b1, w2, b2):
    z1 = X @ w1.T + b1
    a1 = relu(z1)
    z2 = a1 @ w2.T + b2
    a2 = plt_softmax(z2)
    return a2
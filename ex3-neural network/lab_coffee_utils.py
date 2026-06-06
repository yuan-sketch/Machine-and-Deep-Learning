import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from lab_utils_common import dlc


def load_coffee_data():
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5
    X[:, 0] = X[:, 0] * (285 - 150) + 150
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1
    return (X, Y.reshape(-1, 1))


def plt_roast(X, Y):
    Y = Y.reshape(-1, )
    fig, ax = plt.subplots(1, 1)

    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=70, marker='x', c='red', label="Good Roast")
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o', facecolors='none',
               edgecolors='#0D5BDC', linewidth=1, label="Bad Roast")

    tr = np.linspace(175, 260, 50)
    ax.plot(tr, (-3 / 85) * tr + 21, color='#7030A0', linewidth=1)
    ax.axhline(y=12, color='#7030A0', linewidth=1)
    ax.axvline(x=175, color='#7030A0', linewidth=1)

    ax.set_title("Coffee Roasting", size=16)
    ax.set_xlabel("Temperature (Celsius)", size=12)
    ax.set_ylabel("Duration (minutes)", size=12)
    ax.legend(loc='upper right')
    plt.show()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plt_prob(ax, fwb):
    x0_space = np.linspace(150, 285, 40)
    x1_space = np.linspace(11.5, 15.5, 40)
    tmp_x0, tmp_x1 = np.meshgrid(x0_space, x1_space)
    z = np.zeros_like(tmp_x0)

    for i in range(tmp_x0.shape[0]):
        for j in range(tmp_x0.shape[1]):
            x = np.array([[tmp_x0[i, j], tmp_x1[i, j]]])
            z[i, j] = fwb(x)

    cmap = plt.get_cmap('Blues')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    pcm = ax.pcolormesh(tmp_x0, tmp_x1, z,
                        norm=colors.Normalize(vmin=0, vmax=1),
                        cmap=new_cmap, shading='nearest', alpha=0.9)
    ax.figure.colorbar(pcm, ax=ax)


def plt_layer(X, Y, W1, b1, norm_l):
    import tensorflow as tf
    sigmoid = tf.keras.activations.sigmoid

    Y = Y.reshape(-1, )
    fig, ax = plt.subplots(1, W1.shape[1], figsize=(16, 4))

    for i in range(W1.shape[1]):
        layerf = lambda x: sigmoid(np.dot(norm_l(x), W1[:, i]) + b1[i])
        plt_prob(ax[i], layerf)

        ax[i].scatter(X[Y == 1, 0], X[Y == 1, 1], s=70, marker='x', c='red', label="Good Roast")
        ax[i].scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o', facecolors='none',
                      edgecolors='#0D5BDC', linewidth=1, label="Bad Roast")

        tr = np.linspace(175, 260, 50)
        ax[i].plot(tr, (-3 / 85) * tr + 21, color='#7030A0', linewidth=2)
        ax[i].axhline(y=12, color='#7030A0', linewidth=2)
        ax[i].axvline(x=175, color='#7030A0', linewidth=2)
        ax[i].set_title(f"Layer 1, unit {i}")
        ax[i].set_xlabel("Temperature (Celsius)")

    ax[0].set_ylabel("Duration (minutes)")
    plt.show()


def plt_network(X, Y, netf):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    Y = Y.reshape(-1, )

    plt_prob(ax[0], netf)
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], s=70, marker='x', c='red', label="Good Roast")
    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o', facecolors='none',
                  edgecolors='#0D5BDC', linewidth=1)
    ax[0].plot(X[:, 0], (-3 / 85) * X[:, 0] + 21, color='#7030A0')
    ax[0].axhline(y=12, color='#7030A0')
    ax[0].axvline(x=175, color='#7030A0')
    ax[0].set_title("network probability")
    ax[0].legend()

    fwb = netf(X)
    yhat = (fwb > 0.5).astype(int)
    ax[1].scatter(X[yhat[:, 0] == 1, 0], X[yhat[:, 0] == 1, 1], s=70, marker='x', c='orange', label="Predicted Good")
    ax[1].scatter(X[yhat[:, 0] == 0, 0], X[yhat[:, 0] == 0, 1], s=100, marker='o', facecolors='none',
                  edgecolors='#0D5BDC')
    ax[1].set_title("network decision")
    ax[1].legend()
    plt.show()


def plt_output_unit(W, b):
    import tensorflow as tf
    sigmoid = tf.keras.activations.sigmoid

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_ = y_ = z_ = np.linspace(0, 1, 10)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    d = np.zeros_like(x)

    for i in range(10):
        for j in range(10):
            for k in range(10):
                v = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                d[i, j, k] = sigmoid(np.dot(v, W[:, 0]) + b).numpy()

    pcm = ax.scatter(x, y, z, c=d, cmap='Blues')
    ax.set_xlabel("unit 0")
    ax.set_ylabel("unit 1")
    ax.set_zlabel("unit 2")
    ax.view_init(30, -120)
    ax.figure.colorbar(pcm)
    ax.set_title("Layer 2, output unit")
    plt.show()
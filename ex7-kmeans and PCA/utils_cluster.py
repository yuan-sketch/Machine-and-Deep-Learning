# 袁超
# 开发时间：2026/4/15 17:40
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import cv2
from scipy.io import loadmat


def load_data():
    """
    加载K-means作业示例数据集 (二维数据)
    Returns:
        X (array): 样本数据
    """
    data = loadmat("data/ex7data2.mat")
    X = data['X']
    return X


def draw_line(p1, p2, style="-k", linewidth=1):
    """绘制两点之间的直线"""
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)


def plot_data_points(X, idx, K):
    """
    绘制聚类后的散点图，不同簇用不同颜色
    Args:
        X: 样本数据
        idx: 每个样本的簇索引
        K: 聚类数量
    """
    color = plt.cm.rainbow(np.linspace(0, 1, K))
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap=ListedColormap(color))


def plot_progress_k_means(X, centroids, previous, idx, K, i):
    """
    绘制K-means迭代过程：中心点移动轨迹
    """
    plot_data_points(X, idx, K)

    # 绘制中心点
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', c='k', s=200, linewidths=3)

    # 绘制中心点移动轨迹
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous[j, :])

    plt.title(f"Iteration number {i + 1}")


def run_k_means(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    K-means完整迭代流程
    Args:
        X: 样本数据
        initial_centroids: 初始化中心点
        max_iters: 最大迭代次数
        plot_progress: 是否绘制迭代过程
    Returns:
        centroids: 最终中心点
        idx: 样本簇索引
    """
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    plt.figure(figsize=(8, 6))

    for i in range(max_iters):
        print(f"K-Means iteration {i + 1}/{max_iters}")

        # 1. 分配样本到簇
        idx = find_closest_centroids(X, centroids)

        # 绘制迭代过程
        if plot_progress:
            plot_progress_k_means(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # 2. 更新中心点
        centroids = compute_centroids(X, idx, K)

    if plot_progress:
        plt.show()

    return centroids, idx


def find_closest_centroids(X, centroids):
    """
    为每个样本找到最近的中心点
    """
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m, dtype=int)

    for i in range(m):
        distances = np.sum((X[i] - centroids) ** 2, axis=1)
        idx[i] = np.argmin(distances)

    return idx


def compute_centroids(X, idx, K):
    """
    计算每个簇的新中心点
    """
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)

    return centroids


def k_means_init_centroids(X, K):
    """
    随机初始化K-means中心点（从样本中随机选择）
    """
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids


# 图像压缩相关函数
def load_image(path):
    """加载图像并归一化"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img


def plot_image(image):
    """显示图像"""
    plt.imshow(image)
    plt.axis('off')


def plot_k_means_image(X_compressed, idx, centroids, K):
    """
    绘制K-means压缩后的图像
    """
    X_recovered = centroids[idx.astype(int), :]

    # 还原图像尺寸
    n = int(np.sqrt(X_recovered.shape[0]))
    X_recovered = np.reshape(X_recovered, (n, n, 3))

    plot_image(X_recovered)
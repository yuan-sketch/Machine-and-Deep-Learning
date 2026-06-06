# 袁超
# 开发时间：2026/4/15 17:37
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
from utils_cluster import *

'''centroids = k_means_init_centroids(X,K)
for iter in range(iterations):
    idx = find_closest_centroids(X,centroids)#将每个训练实例分配给最近的中心点
    centroids = compute_centroids(X,idx,K)#使用分配给它的点重新计算每个中心点的平均值
'''
def find_closest_centroids(X,centroids):
    #设置K
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0],dtype=int)
    #核心部分
    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i]-centroids[j])#计算元素和中心之间的距离
            distance.append(norm_ij)

        idx[i] = np.argmin(distance)#计算最短距离
    return idx
#用示例数据集检验一下实现情况
#X = load_data()
#print("X的前五个元素是\n",X[:5])
#print("X的shape是\n",X.shape)

#选择一组起始中心
#initial_centroids = np.array([[3,3],[6,2],[8,5]])
#K = 3
#max_iters = 10
#选择最近中心
#idx = find_closest_centroids(X,initial_centroids)
#print("idx的前三个元素\n",idx[:3])
#计算中心点平均值
def compute_centroids(X,idx,K):
    m,n = X.shape#m=样本数量，n=每个样本的特征数
    centroids = np.zeros((K,n))#创建一个K行n列的空数组，用来存放质心
    for i in range(K):
        points = X[idx==i]#取出所有第i族的点
        centroids[i] = np.mean(points,axis=0)#计算这些点的平均值
    return centroids

#检验compute_centroids函数
#K = 3
#centroids = compute_centroids(X,idx,K)
#print("中心是\n",centroids)

#在样本训练集上运行K-means
def run_KMeans(X,initial_centroids,max_iters=10,plot_progress=False):
    m,n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        print("K_Means迭代次数 %d/%d"%(i,max_iters))
        idx = find_closest_centroids(X,centroids)
        if plot_progress:
            plot_progress_k_means(X,centroids,previous_centroids,idx,K,i)
            previous_centroids = centroids
        centroids = compute_centroids(X,idx,K)
    plt.show()
    return centroids,idx

#centroids,idx = run_KMeans(X,initial_centroids,max_iters,plot_progress=True)

#随机初始化
def KMeans_init_centroids(X,K):
    randix = np.random.permutation(X.shape[0])#随机打乱例子的索引
    centroids = X[randix[:K]]
    return centroids

#压缩图片：把图片压缩成16种颜色
#加载图片
original_img = plt.imread('data/bird_small.png')
#plt.imshow(original_img)

print("original_img的shape:",original_img.shape)#前两列表示一个像素的位置，最后一列表示红色绿色或者蓝色
#创建一个m*3的像素颜色矩阵
#确保所有值在0-1之间
original_img = original_img/255#(255,0,0)转换成(1.0,0.0,0.0)
X_img = np.reshape(original_img,(original_img.shape[0]*original_img.shape[1],3))
#(128,128,3)=(163844,3)
K = 16
max_iters = 10
#找出随机化中心
initial_centroids = KMeans_init_centroids(X_img,K)
#用K_Means训练颜色聚类
centroids,idx = run_KMeans(X_img,initial_centroids,max_iters)
#用聚类结果把图片压缩成16色
X_re = centroids[idx,:]
#恢复成图片形状
X_re = np.reshape(X_re,original_img.shape)

fig,ax = plt.subplots(1,2,figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('original')
ax[0].set_axis_off()

ax[1].imshow(X_re*255)
ax[1].set_title("用%d种颜色压缩图片"%K)
ax[1].set_axis_off()

plt.show()

# 袁超
# 开发时间：2026/4/20 15:54
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#生成带噪声的正弦曲线
np.random.seed(42)
x = np.sort(np.random.rand(200,1)*10,axis=0)
y = np.sin(x).ravel()+np.random.randn(200)*0.2

n_trees = 16
learning_rate = 0.3
trees = []
F = np.full_like(y,y.mean())     #用均值平均

fig, axes = plt.subplots(4,4,figsize=(16,9))
mse_history = []

for i in range(n_trees):
    #计算残差
    residual = y-F
    #用一颗浅树拟合残差
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(x,residual)
    pred = tree.predict(x)
    trees.append(tree)

    #更新模型
    F+=learning_rate*pred

    #可视化
    mse = np.mean((y-F)**2)
    mse_history.append(mse)
    ax = axes[i//4][i%4]
    ax.scatter(x,y,s=10,alpha=0.4,color='steelblue',label='真实数据')
    ax.plot(x,F,color='tomato',linewidth=2,label=f'GBOT预测')
    ax.set_title(f'第{i+1}棵树后|MSE={mse:.4f}',fontsize=12)
    ax.legend(fontsize=9)
plt.suptitle('手搓GBDT：逐步逼近真实函数',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_trees+1), mse_history, marker='o', linewidth=2, color='darkred', markersize=6)
plt.xlabel("迭代次数（树的数量）", fontsize=12)
plt.ylabel("MSE（均方误差）", fontsize=12)
plt.title("GBDT 训练过程中 MSE 变化曲线", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
# 袁超
# 开发时间：2026/4/22 17:20
import numpy as np
#内置数据集加载
from sklearn.datasets import load_iris            # 鸢尾花（3分类）
from sklearn.datasets import load_breast_cancer   # 乳腺癌（二分类）
from sklearn.datasets import load_digits          # 手写数字
from sklearn.datasets import make_blobs           # 生成聚类数据
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
#数据划分（训练集和测试集）
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#数据预处理
#标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))         # 默认就是0~1
X_train = scaler.fit_transform(X_train)             # 2. 训练+转换训练集（只在训练集学习最大/最小值）
X_test = scaler.transform(X_test)                   # 3. 测试集只转换、不fit！！


#机器学习模型
#分类模型
#线性回归
from sklearn.linear_model import LinearRegression
# 2.标准化（多特征建议加）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 3.建模训练
lr = LinearRegression()
lr.fit(X_train, y_train)
# 4.预测
y_pred = lr.predict(X_test)
# 5.回归评估指标
print("均方误差MSE：", mean_squared_error(y_test, y_pred))
print("决定系数R2：", r2_score(y_test, y_pred))
lr.coef_   # 各特征权重 w
lr.intercept_ # 截距 b

from sklearn.linear_model import Ridge     # L2正则
model = Ridge(
    alpha=1.0,   # 正则强度，越大正则越强、越不易过拟合
    random_state=42
)

from sklearn.linear_model import Lasso     # L1正则
lasso = Lasso(
    alpha=0.1,      # 正则强度
    max_iter=10000, # 迭代次数，防止不收敛
    random_state=42
)
# 2. 线性正则模型【必须标准化】
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
# 3. 训练Lasso
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train, y_train)
# 4. 预测+评估
y_pred = lasso.predict(X_test)
print("R2 决定系数：", r2_score(y_test, y_pred))
print("MSE 均方误差：", mean_squared_error(y_test, y_pred))
# 查看系数：大量值为0，就是被筛除的特征
print("特征权重：\n", lasso.coef_)

from sklearn.linear_model import ElasticNet # L1+L2
enet = ElasticNet(
    alpha=1.0,        # 总正则化强度（越大正则越强）
    l1_ratio=0.5,     # L1正则占比（0~1）
    max_iter=10000,   # 必加，防止不收敛
    random_state=42
)
# 2. 必须标准化（线性正则模型硬性要求）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 3. 初始化弹性网
enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
enet.fit(X_train, y_train)
# 4. 预测 & 评估
y_pred = enet.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
# 查看权重（部分为0，实现特征筛选）
print("特征系数：", enet.coef_)


# 逻辑回归（二分类神器）
from sklearn.linear_model import LogisticRegression
LogisticRegression(
    penalty="l2",     # 正则：l1 / l2 / none
    C=1.0,            # 正则强度：C越小，正则越强
    solver="lbfgs",   # 优化器
    max_iter=1000,    # 最大迭代次数（防止收敛警告）
    random_state=42
)
# 2.逻辑回归【必须标准化】
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
# 3.建模+训练
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train, y_train)
# 4.预测
y_pred = lr.predict(X_test)       # 类别 0/1
y_prob = lr.predict_proba(X_test)# 两类概率 [p0,p1]
# 5.评估
print(accuracy_score(y_test, y_pred))


# SVM 支持向量机
from sklearn.svm import SVC
'''
1. kernel 核函数（最关键）
linear：线性核（线性可分数据）
poly：多项式核
rbf：高斯核（默认，最常用，万能非线性）
sigmoid：S 型核
2. C 惩罚系数
C 越大：对误分类惩罚越大，越容易过拟合
C 越小：惩罚越小，越简单，可能欠拟合
3. gamma (RBF 核专用)
gamma 越大：模型越复杂，过拟合
gamma 越小：模型越简单，欠拟合
'''
# 1. 标准化（必须！）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 2. 创建SVM模型
svm = SVC(kernel='rbf', C=1, gamma='scale')
# 3. 训练
svm.fit(X_train, y_train)
# 4. 预测
y_pred = svm.predict(X_test)


#贝叶斯网络
from sklearn.naive_bayes import GaussianNB   #高斯朴素贝叶斯————无正则、无复杂参数，开箱即用
# 初始化模型，无参数
gnb = GaussianNB()
# 训练 + 预测
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
# 评估
print("准确率：", accuracy_score(y_test, y_pred))
# 每个类别的预测概率
prob = gnb.predict_proba(X_test)

from sklearn.naive_bayes import MultinomialNB #文本/计数专用
from sklearn.feature_extraction.text import CountVectorizer # 文本转词频
mnb = MultinomialNB(alpha=1.0)
#alpha：平滑系数（拉普拉斯平滑）
#作用：防止特征概率为 0（避免某个单词没出现就判定概率为 0），默认 1.0，几乎不用调参
# 1. 示例：文本数据
texts = ["我喜欢你", "我爱机器学习", "垃圾广告", "退款诈骗"]
labels = [0,0,1,1] # 0=正常 1=垃圾
# 2. 文本转计数特征（核心！必须转成计数矩阵）
vec = CountVectorizer()
X = vec.fit_transform(texts).toarray()
# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
# 4. 训练多项式朴素贝叶斯
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
# 5. 预测与评估
y_pred = model.predict(X_test)
print("准确率：", accuracy_score(y_test, y_pred))


# K近邻
from sklearn.neighbors import KNeighborsClassifier
## 2.必须标准化！KNN强依赖距离
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
## 3.初始化模型
knn = KNeighborsClassifier(n_neighbors=3)
## 4.拟合+预测
knn.fit(X_train, y_train)   # 保存训练数据
y_pred = knn.predict(X_test)
## 5.评估
print(accuracy_score(y_test, y_pred))

#简易神经网络————多层感知分类器
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),  # 隐藏层结构：(神经元数,) 单层 / (64,32) 双层
    activation='relu',          # 激活函数：默认relu（最优），可选logistic/tanh
    solver='adam',              # 优化器：adam（默认，适合大数据）/ lbfgs（小数据）
    alpha=0.0001,               # L2正则化，防止过拟合
    max_iter=1000,              # 最大迭代次数（必须调大，否则不收敛）
    random_state=42
)
# 2. 必须标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 3. 构建MLP模型
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=2000,
    random_state=42
)
# 4. 训练 + 预测
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
# 5. 评估
print("分类准确率：", accuracy_score(y_test, y_pred))


# 决策树
from sklearn.tree import DecisionTreeClassifier
'''
criterion
gini：基尼系数（默认，计算更快）
entropy：信息熵，分裂更精细
max_depth
限制树深度，防止过拟合（最常用剪枝手段）
无需要标准化、归一化，对数值 / 类别特征都友好
'''
tree = DecisionTreeClassifier(
    criterion="gini",      # 划分准则：gini / entropy
    max_depth=None,         # 树最大深度，限制过拟合
    min_samples_split=2,   # 分裂节点最少样本数
    min_samples_leaf=1,     # 叶子节点最少样本数
    random_state=42
)
# 2.初始化模型
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
# 3.训练+预测
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
# 4.评估
print("准确率：", accuracy_score(y_test, y_pred))


# 随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,   # 森林里有多少棵树（默认100）
    criterion="gini",    # 划分准则：gini / entropy
    max_depth=5,        # 每棵树最大深度（防过拟合）
    random_state=42
)
# 2. 创建随机森林
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# 3. 训练
rf.fit(X_train, y_train)
# 4. 预测
y_pred = rf.predict(X_test)
# 5. 评分
print("准确率：", accuracy_score(y_test, y_pred))
#6.看特征重要性
print(rf.feature_importances_)
#GBDT————串行，树之间依赖，逐棵纠错
from sklearn.ensemble import GradientBoostingClassifier   #Gradient Boosting Decision Tree
gbdt = GradientBoostingClassifier(
    n_estimators=100,   # 决策树的数量
    learning_rate=0.1,  # 学习率（越小训练越慢，越稳定）
    max_depth=3,        # 单棵树深度（防过拟合）
    random_state=42
)
# 2. 初始化GBDT模型
gbdt = GradientBoostingClassifier(
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
# 3. 训练 + 预测
gbdt.fit(X_train, y_train)
y_pred = gbdt.predict(X_test)
# 4. 评估
print("准确率：", accuracy_score(y_test, y_pred))
#XGBoost————优化版GBDT
# 分类任务专用
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=100,    # 决策树数量
    max_depth=3,         # 树最大深度（防过拟合）
    learning_rate=0.1,   # 学习率（越小越精准，训练越慢）
    subsample=0.8,       # 随机采样80%样本训练（防过拟合）
    colsample_bytree=0.8,# 随机采样80%特征（防过拟合）
    random_state=42
)#学习率调小 + 树数量调大，精度最高！
# 2. 初始化XGBoost模型
xgb = XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
# 3. 训练 + 预测
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
# 4. 评估
print("XGBoost 准确率：", accuracy_score(y_test, y_pred))
#LightGBM————XGBoost 轻量化极速版，梯度提升树优化算法
# 分类任务专用
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(
    n_estimators=100,    # 树的数量
    learning_rate=0.1,   # 学习率
    max_depth=3,         # 树深度（防过拟合）
    num_leaves=31,       # 叶子节点数（LightGBM核心参数），控制树复杂度，小数据设小一点，大数据默认即可
    random_state=42
)
# 2. 初始化LightGBM模型
lgb = LGBMClassifier(
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
# 3. 训练 + 预测
lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)
# 4. 评估
print("LightGBM 准确率：", accuracy_score(y_test, y_pred))

# PCA降维
from sklearn.decomposition import PCA
pca = PCA(
    n_components=2,   # 降到几维（最常用）
)
# 2.必须标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 3.PCA降维：4维 → 2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# 4.查看保留信息占比
print("各主成分方差占比：", pca.explained_variance_ratio_)
print("累计保留信息：", sum(pca.explained_variance_ratio_))
# 5.降维后可视化
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.title("PCA 降维可视化")
plt.show()
pca.explained_variance_ratio_  # 每个主成分的信息贡献率
pca.components_                # 投影矩阵



# KMeans聚类
from sklearn.cluster import KMeans
kmeans = KMeans(
    n_clusters=3,  # 聚成几类（最核心！）
    random_state=42
)
# 2. 必须标准化（KMeans基于距离）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 3. KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)  # 直接得到聚类结果
# 4. 查看结果
print("聚类标签：", labels)
print("中心点：", kmeans.cluster_centers_)

# 画聚类图————用前两个特征画图
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='red', marker='*', label='Centers')
plt.title("KMeans 聚类")
plt.legend()
plt.show()

kmeans.labels_           # 每个样本的簇标签
kmeans.cluster_centers_  # 簇中心点坐标
kmeans.inertia_          # 总距离平方和（评估好坏）

#手肘法选K
inertias = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K, inertias, 'o-')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("手肘法选K")
plt.show()


#DBSCAN密度聚类————不需要指定聚类个数 K，自动识别噪声 / 异常值
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(
    eps=0.5,          # 邻域半径
    min_samples=5,    # 最小邻居数
    metric='euclidean'# 距离度量，默认欧氏距离
)#必须标准化
# 1. 标准化（必做）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 2. DBSCAN 训练
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
# 3. 标签说明
# -1 = 噪声/异常点
print("聚类标签：", set(labels))
# 4. 简单绘图
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.show()
dbscan.labels_       # 每个样本聚类标签，-1 为噪声


#AgglomerativeClustering层次聚类
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(
    n_clusters=3,        # 最终聚类数【必须指定】
    linkage="ward",      # 簇间距离计算方式（核心）
    metric="euclidean"   # 样本距离，默认欧氏距离
)
'''
linkage 四种常用
ward（默认）：合并后簇内方差最小，最稳定、最常用
single：两簇最近样本距离
complete：两簇最远样本距离
average：两簇所有样本平均距离
'''
# 1. 标准化（必做）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 2. 层次聚类
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(X_scaled)
# 3. 可视化
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.title("层次聚类")
plt.show()


#SelectKBest特征筛选————有监督，保留原始特征，只做筛选
'''
逐个计算单个特征 与 标签 y 的统计打分
按分数从高到低排序
保留分数最高的前 k 个特征，其余删除
'''
from sklearn.feature_selection import SelectKBest,f_classif,f_regression
# 分类场景
selector = SelectKBest(score_func=f_classif, k=5)
# 回归场景
selector = SelectKBest(score_func=f_regression, k=5)
'''
场景	打分函数	             说明
分类	f_classif	         方差分析 F 值，默认、最常用
分类	chi2	             卡方检验，只允许非负特征（词频、计数）
分类	mutual_info_classif	 互信息，捕捉非线性关系
回归	f_regression	     线性回归 F 检验
回归	mutual_info_regression	回归非线性互信息
'''
# 2. 初始化：选前4个最优特征
selector = SelectKBest(score_func=f_classif, k=4)
# 3. 拟合筛选（只用训练集，防止数据泄露）
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new  = selector.transform(X_test)
# 4. 查看特征得分 & 被选中/剔除
print("各特征得分：", selector.scores_)
print("被选中特征掩码：", selector.get_support())
# 5. 用筛选后的特征建模
model = RandomForestClassifier(random_state=42)
model.fit(X_train_new, y_train)
selector.scores_      # 每个特征的评分
selector.pvalues_     # 显著性p值
selector.get_support()# 布尔数组，True=选中特征


#IsolationForest孤立森林，异常检测
from sklearn.ensemble import IsolationForest
ifo = IsolationForest(
    n_estimators=100,   # 决策树数量（默认100）
    contamination=0.1,   # 污染率：数据中异常值的比例（核心参数）
    random_state=42
)
# 1. 生成测试数据（含正常点+异常点）
X = np.random.randn(100, 2)  # 正常数据
X = np.vstack((X, np.array([[5,5],[6,5],[5,6]]))) # 加入3个异常点
# 2. 初始化孤立森林
ifo = IsolationForest(contamination=0.05, random_state=42)
# 3. 训练+预测
labels = ifo.fit_predict(X)
# 4. 查看结果
print("聚类标签(1正常,-1异常)：", np.unique(labels))
print("异常点数量：", sum(labels == -1))
# 5. 可视化
plt.scatter(X[:,0], X[:,1], c=labels, cmap="coolwarm")
plt.title("孤立森林 异常检测")
plt.show()




#训练+预测
model.fit(X_train, y_train)   # 训练
y_pred = model.predict(X_test)# 预测
y_score = model.predict_proba(X_test)[:,1] # 预测概率（画ROC用）

#模型评价指标
from sklearn.metrics import *

accuracy_score(y_test, y_pred)   # 准确率
confusion_matrix(y_test, y_pred) # 混淆矩阵
classification_report(y_test, y_pred) # 精确率、召回率、F1
roc_auc_score(y_test, y_score)   # AUC值
roc_curve(y_test, y_score)       # ROC曲线坐标





#pandas库
import pandas as pd

#读取csv文件
df = pd.read_csv("文件名.csv")

#保存为csv文件
df.to_csv("保存文件名.csv",index=False)

#数据快速查看
df.head()           #查看前五行（numpy数组不能用）
df.tail()           #查看最后五行
df.shape            #输出行数、列数
df.columns          #查看所有列名
df.info()           #查看数据类型、损失值
df.describe()       #数值列统计：均值、方差、最大、最小

df["列名"]           # 选取单列
df[["列1","列2"]]    # 选取多列

# 举例：筛选欺诈样本 Class=1
df[df["Class"] == 1]
# 筛选正常样本
df[df["Class"] == 0]

df.loc[行标签, 列名]    # 按标签取值
df.iloc[行下标, 列下标] # 按位置取值（数字索引）

# 统计标签各类数量（信用卡、乳腺癌数据集必备）
df["Class"].value_counts()

df.isnull()           # 判断每个位置是否缺失
df.isnull().sum()     # 统计每列缺失值总数
df.dropna()           # 删除含缺失值的行
df.fillna(0)          # 缺失值填充为0

# 新增一列
df["新列名"] = 数据

# 删除指定列
df = df.drop("列名", axis=1)

df.astype(int)        # 强制转换数据类型
df.drop_duplicates()  # 删除重复行
df.rename(columns={"旧名":"新名"}) # 重命名列






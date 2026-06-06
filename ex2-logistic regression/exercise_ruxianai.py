# 袁超
# 开发时间：2026/4/22 15:32
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


bc = load_breast_cancer()
x = bc.data
y = bc.target
#1.划分数据集
x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.2,stratify=y
)

#2.数据标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#3.训练逻辑回归模型
model = LogisticRegression()
model.fit(x_train_scaled,y_train)

print(f'测试集的准确率:{model.score(x_test_scaled,y_test,sample_weight=None):.4f}')

#4.预测概率
y_score = model.predict_proba(x_test_scaled)[:,1]
#5.计算ROC坐标
fpr ,tpr,threshold = roc_curve(y_test,y_score)
auc = roc_auc_score(y_test,y_score)

#6.画图
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'Roc curve(AUC={auc:.3f})',linewidth=2)
plt.plot([0,1],[0,1],'k--',label='RandomGuess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Roc Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
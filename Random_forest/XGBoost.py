# 袁超
# 开发时间：2026/4/20 16:22
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#加载加州房价数据集
price = fetch_california_housing()
x,y = price.data,price.target
feature_names = price.feature_names

#划分训练集和测试集
x_train,x_test,y_train,y_test= train_test_split(
    x,y,test_size=0.2,random_state=42
)

print(f"训练集大小：{x_train.shape}")
print(f"测试集大小：{x_test.shape}")
'''
#随机森林
start = time.time()
rf =RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1)
rf.fit(x_train,y_train)
rf_time = time.time()-start
rf_pred = rf.predict(x_test)

#---XGBoost---
start = time.time()
xgb = XGBRegressor(
    n_estimators=200,         #决策树数量
    max_depth=5,              #每棵树的最大深度
    learning_rate=0.1,        #学习率
    subsample=0.8,            #每次训练用多少比例的样本
    colsample_bytree=0.8,     #每次训练利用多少比例的特征
    random_state=42,          #固定随机种子
    n_jobs=-1                 #用满cpu核心加速
)

xgb.fit(x_train,y_train)
xgb_time = time.time()-start
xgb_pred = xgb.predict(x_test)

#结果对比
print("="*66)
print(f"{'指标':<15}{'随机森林':>12}{'XGBoost':>12}")
print("="*66)
print(f"{'MSE':<15}{mean_squared_error(y_test,rf_pred):>12.4f}{mean_squared_error(y_test,xgb_pred):>12.4f}")
print(f"{'R2 Score':<15}{r2_score(y_test,rf_pred):>12.4f}{r2_score(y_test,xgb_pred):>12.4f}")
print(f"{'训练时间(秒)':<15}{rf_time:>12.2f}{xgb_time:>12.4f}")
'''

#XGBoost通过Early Stopping防止过拟合
xgb_es = XGBRegressor(
    n_estimators=10000,      #设置一个很大的上限
    max_depth=5,
    learning_rate=0.05,      #用更小的学习率
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20 #20轮不提升就停止
)

xgb_es.fit(
    x_train,y_train,
    eval_set=[(x_test,y_test)],
    verbose = 50             #每50轮打印一次
)
xgb_es_pred = xgb_es.predict(x_test)
print(f"\n最佳迭代轮次：{xgb_es.best_iteration}")
print(f"最佳MSE：{mean_squared_error(y_test,xgb_es_pred):>7.4f}")

#获取训练过程的损失变化
results = xgb_es.evals_result()

plt.figure(figsize=(10,5))
plt.plot(results['validation_0']['rmse'],
         color='tomato',linewidth=2)
plt.xlabel('迭代轮数(树的数量)')
plt.ylabel('RMSE')
plt.title('XGBoost 训练过程')
plt.axvline(x=xgb_es.best_iteration,color='green',
            linestyle='--',
            label=f'最佳轮数={xgb_es.best_iteration}')
plt.legend()
plt.grid(True,alpha =0.3)
plt.show()
# 袁超
# 开发时间：2026/4/7 16:37
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost_logistic_reg(X,y,w,b,lambda_=1):
    m,n=X.shape
    cost=0.
    for i in  range(m):
        z_i=np.dot(X[i],w)+b
        f_wb_i=sigmoid(z_i)
        cost+=-y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)

    cost=cost/m

    reg_cost=0
    for j in range(n):
        reg_cost+=(w[j]**2)
    reg_cost=(lambda_/(2*m))*reg_cost

    total_cost=cost+reg_cost
    return total_cost

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)


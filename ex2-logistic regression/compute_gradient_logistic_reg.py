# 袁超
# 开发时间：2026/4/7 17:07
import numpy as np
def compute_gradient_logistic_reg(X, y, w, b, lambda_):

    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.0  # scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_db, dj_dw



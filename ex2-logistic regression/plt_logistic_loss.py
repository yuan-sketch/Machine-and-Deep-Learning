# 袁超
# 开发时间：2026/4/7 14:03
"""----------------------------------------------------------------
 logistic_loss plotting routines and support
"""

from matplotlib import cm
from lab_utils_common import sigmoid, dlblue, dlorange, np, plt, compute_cost_matrix

def compute_cost_logistic_sq_err(X, y, w, b):
    """
    compute sq error cost on logicist data1 (for negative example only, not used in practice)
    Args:
      X (ndarray): Shape (m,n) matrix of examples with multiple features
      w (ndarray): Shape (n)   parameters for prediction
      b (scalar):              parameter  for prediction
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)                 #add sigmoid to normal sq error cost for linear regression
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return np.squeeze(cost)

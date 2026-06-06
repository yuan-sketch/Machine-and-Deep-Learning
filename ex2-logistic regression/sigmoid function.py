# 袁超
# 开发时间：2026/4/6 21:40
import numpy as np
def sigmoid(z):

    g = 1 / (1 + np.exp(-z))

    return g
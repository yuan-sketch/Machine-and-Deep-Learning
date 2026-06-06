# 袁超
# 开发时间：2026/4/6 13:23
import numpy as np

def load_data():
    data = np.loadtxt("data1/ex1data1.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

def load_data_multi():
    data = np.loadtxt("data1/ex1data2.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y
# 袁超
# 开发时间：2026/5/6 17:30
import pandas as pd
import torch
from torch import nn
from my_utils import semilogy

#读取数据集
train_data = pd.read_csv('../house_price/train.csv')
test_data = pd.read_csv('../house_price/test.csv')

all_features = pd.concat((train_data.iloc[:,1:-1],
                         test_data.iloc[:,1:]))

#预处理数据集——正态分布标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
#标准化后，每个特征均值变为0，所以可以用0来代替缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 在执行 get_dummies 之前，检查每个 Object 列的唯一值数量
object_columns = all_features.select_dtypes(include=['object']).columns
total_dummy_cols = 0
for col in object_columns:
    # 唯一值数量 + 1 (因为 dummy_na=True)
    count = all_features[col].nunique() + 1
    total_dummy_cols += count

# 理论总列数 = 数值特征列数 + total_dummy_cols
print(f"数值特征列数: {len(numeric_features)}")
print(f"分类特征生成的理论列数: {total_dummy_cols}")

#dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
#存在一列看不见的object
all_features = all_features.astype(float)
print(all_features.shape)
print(train_data.shape,test_data.shape,'\n',train_data.iloc[0:4,])

#转换成tensor类型
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values,
                            dtype=torch.float32).view(-1,1)

#训练模型
#使用一个基本的线性回归模型和平方差损失函数
loss = torch.nn.MSELoss()
def get_net(feature_sum):
    net = nn.Linear(feature_sum,1)
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)
    return net

#对数均方根误差实现
def log_rmse(net,features,labels):
    with torch.no_grad():
        #将小于1 的值设置成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

def train(net, train_features, train_labels, test_features,
          test_labels, num_epochs, learning_rate, weight_decay,batch_size):
    train_ls = []
    test_ls = []
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    #使用Adam优化器
    optimizer = torch.optim.Adam(params=net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    net =net.float()
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features,test_labels))
    return  train_ls,test_ls

#把数据集分成K份，返回第i份作为验证集，其他K-1份作为训练集
def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1)*fold_size)
        X_part, y_part = X[idx,:],y[idx]
        if j==i:
            X_valid,y_valid = X_part,y_part
        elif X_train is None:
            X_train, y_train = X_part,y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train,y_train,X_valid,y_valid

#在K折交叉验证中训练K次并返回训练和验证的平均误差
def k_fold(k ,X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0,0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs,
                                   learning_rate, weight_decay, batch_size)
        train_l_sum +=train_ls[-1]
        valid_l_sum +=valid_ls[-1]
        if i == 0:
            semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse',
                     range(1, num_epochs+1), valid_ls, ['train','valid'])
        print('fold %d, train rmse %f, valid rmse %f'%(i, train_ls[-1],valid_ls[-1]))
    return train_l_sum/k, valid_l_sum/k

#模型选择
k, num_epochs, lr, weight_decay,batch_size = 10,150,4,0,64
train_l, valid_l = k_fold(k, train_features, train_labels,
                          num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f'%(k, train_l, valid_l))

#预测函数
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net((train_features.shape[1]))
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse')
    print('train rmse %f'%train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(-1,1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv('submission.csv',index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)
# 袁超
# 开发时间：2026/6/9 12:06

# 包含跳字模型和连续词袋模型
# 跳字模型---基于某个词来生成它在文本序列周围的词
# 连续词袋模型---基于某中心词在文本序列前后的背景词来生成该中心词
import collections
import math
import random

import time
import os
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
import sys
sys.path.append("..")
import d2lzh as d2l


# 提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        # 只要句子长度大于等于2，每个词都要做中心词
        centers += st
        for center_i in range(len(st)):
            # 在1和max_window_size之间随机均匀采样一个整数作为背景窗口大小
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i-window_size),
                                 min(len(st), center_i+1+window_size)))
            # 将中心词排除在背景词外
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

# 负采样
# 对于一对中心词和背景词，随机采样背景词个数K倍个噪声词
def get_negatives(all_contexts, sampling_weights, K):
    # all_contexts:每个词的背景词列表
    # sampling_weights每个词词频的0.75次幂
    # K:噪声词相比于背景词个数的倍数
    all_negatives, neg_candidates, i =[], [], 0
    # 词表中词的个数
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 从population随机选取k次数据，返回一个列表
                # 根据每个词的权重随机生成k个次的索引作为噪声词
                # 为了高效运算，可以将k设置的大一些
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5)
                )
            neg, i = neg_candidates[i], i+1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives



# 读取数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(contexts) == len(centers) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
    def __getitem__(self, index):
        return (self.centers[index],
                self.contexts[index],
                self.negatives[index])
    def __len__(self):
        return len(self.centers)

# 小批量读取函数
def batchify(data):
    # 中心词，背景词，噪声词
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [],[],[],[]
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [contexts_negatives+ [0] * (max_len-cur_len)]
        masks += [[1]*cur_len + [0]*(max_len-len(context))]
        labels += [[1]*len(context)+[0]*(max_len-len(context))]
    return (
            torch.tensor(centers).view(-1,1),
            torch.tensor(contexts_negatives),
            torch.tensor(masks),
            torch.tensor(labels)
            )
if __name__ == '__main__':
    # 预处理数据集
    with open('data/ptb/ptb.train.txt', 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]

    for st in raw_dataset[:3]:
        print('# tokens:', len(st), st[:5])

    # 建立词语索引---只保留数据集中至少出现5次的词
    counter = collections.Counter(
        [tk for st in raw_dataset for tk in st]
    )
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    # 将词映射到整数索引
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]
    num_tokens = sum([len(st) for st in dataset])


    # 二次采样---数据集中每个被索引词wi将有一定的概率被丢弃
    def discard(idx):
        # 与均匀分布相比，确实该词是否被剔除
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / counter[idx_to_token[idx]] * num_tokens
        )


    subsampled_dataset = [[tk for tk in st if not discard(tk)]
                          for st in dataset]


    # 比较一个词在二次取样前后出现在数据集中的次数
    def compare_counts(token):
        return '# %s: before=%d, after=%d' % (token, sum(
            [st.count(token_to_idx[token]) for st in dataset]
        ), sum(
            [st.count(token_to_idx[token]) for st in subsampled_dataset]
        ))


    print(compare_counts('the'))
    print(compare_counts('join'))
    # 创建人工数据集，含有词数分别为7和3的两个句子，设最大窗口是2，打印所有中心词和背景词
    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print('dataset:', tiny_dataset)
    for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print('center', center, 'has contexts', context)

    # 实验中，设置最大背景窗口大小为5，提取数据集中所有中心词及其背景词
    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)


    sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]
    all_negatives = get_negatives(all_contexts, sampling_weights, 5)
    batch_size = 512
    num_workers = 4
    dataset = MyDataset(all_centers, all_contexts, all_negatives)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                                collate_fn=batchify,num_workers=4)
    for batch in data_iter:
        for name, data in zip(['centers', 'contexts_negatives',
                               'masks','labels'], batch):
            print(name, 'shape:', data.shape)
        break

# 跳字模型---词典大小为20，词向量维度为4
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
# 嵌入层的输入为词的索引，输出一个词的索引i,嵌入层返回权重矩阵的第i行作为它的词向量
x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.long)
print(embed(x))

# 小批量乘法
X = torch.ones((2,1,4))
Y = torch.ones((2,4,6))
print(torch.bmm(X, Y).shape)

# 跳字模型前向计算
def skip_gram(center, contexts_and_negatives,embed_v, embed_u):
    # 每一个词由背景词向量和中心词向量表示
    # 需要两个嵌入表示
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    # batch、emb、num
    pred = torch.bmm(v, u.permute(0,2,1))
    return pred

# 训练模型---二元交叉熵损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss,self).__init__()
    def forward(selfself, inputs, targets, mask=None):
        inputs, targets = inputs.float(), targets.float()
        mask = mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', weight=mask
        )
        return res.mean(dim=1)

loss = SigmoidBinaryCrossEntropyLoss()

# 通过掩码变量指定小批量中参与损失函数计算的部分预测值和标签：
pred = torch.tensor([[1.5,0.3,-1,2], [1.1,-0.6,2.2,0.4]])
# 标签变量label中的1和0分别表示背景词和噪声词
label = torch.tensor([[1,0,0,0], [1,1,0,0]])
# 掩码变量
mask = torch.tensor([[1,1,1,1], [1,1,1,0]])
print(loss(pred,label,mask)*mask.shape[1] / mask.float().sum(dim=1))


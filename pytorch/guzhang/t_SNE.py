# 袁超
# 开发时间：2026/6/5 13:48

# plot_tsne.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ==========================================
# 1. 网络结构定义
# ==========================================
class VibrationDataset(Dataset):
    def __init__(self, data, labels=None, is_train=False):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.is_train = is_train

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx].clone(), self.labels[idx]


class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out))) + self.shortcut(x)
        return self.relu(out)


class FaultDiagnosisModel(nn.Module):
    def __init__(self, num_classes=5):
        super(FaultDiagnosisModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResBlock1D(32, 64, stride=2)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


# ==========================================
# 2. t-SNE
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_tsne_features(model, data_loader, device, save_path='tsne_fault_clustering.png'):
    model.eval()
    features_all, labels_all = [], []

    print("正在提取高维特征...")
    with torch.no_grad():
        for inputs, labels in data_loader:
            x = model.maxpool(model.relu(model.bn1(model.conv1(inputs.to(device)))))
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            features = model.avgpool(x).view(x.size(0), -1)
            features_all.append(features.cpu().numpy())
            labels_all.append(labels.numpy())

    features_all = np.vstack(features_all)
    labels_all = np.concatenate(labels_all)

    print("开始 t-SNE 降维计算...")
    tsne = TSNE(n_components=2, init='pca', random_state=42, max_iter=1000)
    features_2d = tsne.fit_transform(features_all)

    class_names = ['泵故障 (beng)', '滚动体故障 (gundongti)', '内环故障 (neihuan)', '外环故障 (waihuan)',
                   '正常状态 (zhengchang)']
    plt.figure(figsize=(9, 7.5), dpi=300)
    academic_colors = ['#E64B35FF', '#4DBBD5FF', '#00A087FF', '#3C5488FF', '#F39B7FFF']

    for i in range(5):
        idx = (labels_all == i)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], c=academic_colors[i], label=class_names[i], alpha=0.85,
                    s=30)

    plt.title('基于 1D-CNN + SE 注意力机制的故障特征 t-SNE 聚类图', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('t-SNE 维度 1', fontsize=11, fontweight='bold')
    plt.ylabel('t-SNE 维度 2', fontsize=11, fontweight='bold')
    plt.legend(loc='best', fontsize=10, frameon=True, edgecolor='#D3D3D3')
    plt.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"聚类图已保存至: {save_path}")


if __name__ == "__main__":
    # 1. 读取用于画图的数据 (通常用训练集即可验证聚类效果)
    train_files = [
        'beng/train_424000.csv', 'gundongti/train_424000.csv',
        'neihuan/train_424000.csv', 'waihuan/train_424000.csv', 'zhengchang/train_424000.csv'
    ]
    all_train_data, all_train_labels = [], []
    for label, file_path in enumerate(train_files):
        df = pd.read_csv(file_path)
        signal_data = df.iloc[0:424, 0:1000].values
        all_train_data.append(signal_data)
        all_train_labels.append(np.full(len(signal_data), label))

    X_train_raw = np.vstack(all_train_data)
    y_train = np.concatenate(all_train_labels)

    # 必须执行标准化才能输入模型
    X_train = (X_train_raw - np.mean(X_train_raw, axis=1, keepdims=True)) / (
                np.std(X_train_raw, axis=1, keepdims=True) + 1e-8)

    # 2. 加载模型权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultDiagnosisModel(num_classes=5).to(device)

    weight_path = 'best_model_fold_1.pth'  # 这里选择你想查看的那一折的模型
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print(f"成功加载预训练权重: {weight_path}")
    else:
        print(f"未找到权重文件 {weight_path}，请先运行 train.py")
        exit()

    # 3. 构建 DataLoader 并画图
    vis_dataset = VibrationDataset(X_train, y_train, is_train=False)
    vis_loader = DataLoader(vis_dataset, batch_size=64, shuffle=False)

    plot_tsne_features(model, vis_loader, device, save_path='tsne_report_result.png')
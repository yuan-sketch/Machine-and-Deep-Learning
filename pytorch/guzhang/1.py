# 袁超
# 开发时间：2026/6/5 12:59

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import copy

print(torch.__version__)          # 查看当前的 PyTorch 版本
print(torch.cuda.is_available())  # 看看是否支持 CUDA
# 构建数据集和数据增强
class VibrationDataset(Dataset):
    def __init__(self, data, labels=None, is_train=False):
        """
        data: numpy array, shape (N, 1000)
        labels: numpy array, shape (N,) 或 None
        """
        # 增加通道维度，适配 Conv1d 输入格式 (Batch, Channels, Length) -> (N, 1, 1000)
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone()

        # 训练时应用轻量级数据增强：随机高斯噪声
        if self.is_train:
            if np.random.rand() > 0.5:
                noise = torch.randn_like(x) * 0.01  # 噪声系数可根据实际幅值微调
                x = x + noise

        if self.labels is not None:
            return x, self.labels[idx]
        return x


# 定义模型
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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
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
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class FaultDiagnosisModel(nn.Module):
    def __init__(self, num_classes=5):
        super(FaultDiagnosisModel, self).__init__()
        # 大卷积核提取初始长周期特征
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 堆叠残差块
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 训练与五折交叉验证
def train_and_validate(X_train, y_train, X_test, num_epochs=50, batch_size=64, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    # 用于存储测试集的概率预测，以进行软投票
    test_probs = np.zeros((len(X_test), 5))
    test_dataset = VibrationDataset(X_test, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"--- Starting Fold {fold + 1}/5 ---")

        # 划分本折的训练集和验证集
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        train_loader = DataLoader(VibrationDataset(X_tr, y_tr, is_train=True), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(VibrationDataset(X_va, y_va, is_train=False), batch_size=batch_size, shuffle=False)

        model = FaultDiagnosisModel(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_f1 = 0.0
        best_model_weights = copy.deepcopy(model.state_dict())

        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # 验证阶段
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(labels.numpy())

            # 计算 Macro-F1
            macro_f1 = f1_score(val_targets, val_preds, average='macro')

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_model_weights = copy.deepcopy(model.state_dict())

        print(f"Fold {fold + 1} Best Macro-F1: {best_f1:.4f}")

        # 加载本折最优权重并进行测试集推理
        model.load_state_dict(best_model_weights)
        model.eval()
        models.append(model)

        # 对测试集进行预测并累加概率 (Soft Voting)
        fold_probs = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = torch.softmax(model(inputs), dim=1).cpu().numpy()
                fold_probs.extend(outputs)

        test_probs += np.array(fold_probs) / kf.n_splits

    return models, test_probs


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch

# ==========================================
# 全局设置：确保 matplotlib 完美显示简体中文
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def plot_tsne_features(model, data_loader, device, save_path='tsne_fault_clustering.png'):
    """
    提取模型全连接层（FC）之前的256维特征，并绘制t-SNE降维散点图
    """
    model.eval()
    features_all = []
    labels_all = []

    print("正在从模型中提取高维特征...")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            # 显式前向传播至全连接层之前，获取全局平均池化后的 256 维特征
            x = model.maxpool(model.relu(model.bn1(model.conv1(inputs))))
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.avgpool(x)
            features = x.view(x.size(0), -1)  # 形状: (Batch, 256)

            features_all.append(features.cpu().numpy())
            labels_all.append(labels.numpy())

    # 合并所有批次的数据
    features_all = np.vstack(features_all)
    labels_all = np.concatenate(labels_all)

    print(f"特征提取完毕，特征矩阵形状: {features_all.shape}。开始进行 t-SNE 降维计算...")

    # 初始化 t-SNE
    # 推荐 init='pca'，这样能使高维空间的全局拓扑结构在低维映射中更稳定
    tsne = TSNE(n_components=2, init='pca', random_state=42, n_iter=1000, learning_rate='auto')
    features_2d = tsne.fit_transform(features_all)

    # 严格对应您读取的 5 种工作状态标签 (0-4)
    class_names = [
        '泵故障 (beng)',
        '滚动体故障 (gundongti)',
        '内环故障 (neihuan)',
        '外环故障 (waihuan)',
        '正常状态 (zhengchang)'
    ]

    # 开始构建高分辨率学术图表
    plt.figure(figsize=(9, 7.5), dpi=300)  # 300 DPI 达到期刊无损印刷标准

    # 选用 Nature/Science 风格的高对比度、不刺眼的学术色系
    academic_colors = ['#E64B35FF', '#4DBBD5FF', '#00A087FF', '#3C5488FF', '#F39B7FFF']

    # 循环绘制每一类故障的散点
    for i in range(5):
        idx = (labels_all == i)
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            c=academic_colors[i],
            label=class_names[i],
            alpha=0.85,
            edgecolors='none',
            s=30  # 散点大小
        )

    # 完善图表元素的学术规范
    plt.title('基于 1D-CNN + SE 注意力机制的故障特征 t-SNE 聚类图', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('t-SNE 维度 1', fontsize=11, fontweight='bold')
    plt.ylabel('t-SNE 维度 2', fontsize=11, fontweight='bold')

    # 美化图例与背景网格
    plt.legend(loc='best', fontsize=10, frameon=True, shadow=False, edgecolor='#D3D3D3')
    plt.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')

    # 自动调整布局，防止标签切边
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"高级 t-SNE 聚类图已成功生成并保存至: {save_path}")
# 主程序
import os

if __name__ == "__main__":
    # ==========================================
    # 1. 配置参数与文件路径
    # ==========================================
    train_files = [
        'beng/train_424000.csv',
        'gundongti/train_424000.csv',
        'neihuan/train_424000.csv',
        'waihuan/train_424000.csv',
        'zhengchang/train_424000.csv'
    ]
    test_file = 'test_106000.csv'  # 测试集文件名

    all_train_data = []
    all_train_labels = []

    print("正在从多个文件中加载训练数据...")

    # ==========================================
    # 2. 循环读取 5 个训练文件
    # ==========================================
    for label, file_path in enumerate(train_files):
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，请检查路径。")
            continue

        # 读取文件：pd.read_csv 默认将第1行视为表头
        # 此时 df 的第0行实际上是原文件的第2行
        df = pd.read_csv(file_path)

        # 提取第2行到第425行（在 DataFrame 中对应索引 0 到 423）
        # 假设前1000列是信号数据
        signal_data = df.iloc[0:424, 0:1000].values

        all_train_data.append(signal_data)
        # 生成对应的标签 (0-4)
        all_train_labels.append(np.full(len(signal_data), label))

        print(f"已加载 {file_path}: 标签 {label}, 样本数 {len(signal_data)}")

    # 合并数据
    X_train_raw = np.vstack(all_train_data)  # 形状: (2120, 1000)
    y_train = np.concatenate(all_train_labels)  # 形状: (2120,)

    # ==========================================
    # 3. 读取测试集
    # ==========================================
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        test_ids = test_df.iloc[:, 0].values  # 第一列 ID
        X_test_raw = test_df.iloc[:, 1:1001].values  # 第2列到1001列为信号
        print(f"测试集加载完成: 样本数 {len(X_test_raw)}")
    else:
        print("未找到测试集文件，将跳过推理阶段。")
        X_test_raw = None

    # ==========================================
    # 4. 数据标准化 (Z-score)
    # ==========================================
    print("开始数据标准化...")
    # 训练集标准化
    mean = np.mean(X_train_raw, axis=1, keepdims=True)
    std = np.std(X_train_raw, axis=1, keepdims=True) + 1e-8
    X_train = (X_train_raw - mean) / std

    # 测试集标准化
    if X_test_raw is not None:
        t_mean = np.mean(X_test_raw, axis=1, keepdims=True)
        t_std = np.std(X_test_raw, axis=1, keepdims=True) + 1e-8
        X_test = (X_test_raw - t_mean) / t_std
    else:
        X_test = np.array([])

    # ==========================================
    # 5. 启动模型训练与五折交叉验证
    # ==========================================
    # 调用之前定义的 train_and_validate 函数
    trained_models, test_ensemble_probs = train_and_validate(
        X_train, y_train, X_test,
        num_epochs=50,
        batch_size=64,
        learning_rate=1e-3
    )

    # ==========================================
    # 6. 生成预测结果
    # ==========================================
    if X_test_raw is not None:
        final_predictions = np.argmax(test_ensemble_probs, axis=1)
        submission = pd.DataFrame({
            'id': test_ids,
            'label': final_predictions
        })
        submission.to_csv('fault_detection_results.csv', index=False)
        print("任务完成！预测结果已保存至 'fault_detection_results.csv'。")

    # ==========================================
    # 7.生成 t-SNE 特征聚类图
    # ==========================================
    if len(trained_models) > 0:
        print("\n--- 开始生成学术报告所需的 t-SNE 特征可视化图 ---")

        # 1. 选用训练集数据进行特征展示（注意：关闭数据增强 is_train=False，保证特征纯净）
        vis_dataset = VibrationDataset(X_train, y_train, is_train=False)
        vis_loader = DataLoader(vis_dataset, batch_size=64, shuffle=False)

        # 2. 取出五折交叉验证中训练好的第一个子模型
        best_sub_model = trained_models[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. 调用函数绘制并保存图片
        plot_tsne_features(best_sub_model, vis_loader, device, save_path='tsne_report_result.png')
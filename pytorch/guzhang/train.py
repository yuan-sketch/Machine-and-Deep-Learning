# 袁超
# 开发时间：2026/6/5 14:22

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import copy


# ==========================================
# 1. 数据集与模型架构定义
# ==========================================
class VibrationDataset(Dataset):
    def __init__(self, data, labels=None, is_train=False):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        if self.is_train and np.random.rand() > 0.5:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        if self.labels is not None:
            return x, self.labels[idx]
        return x


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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==========================================
# 2. 训练逻辑
# ==========================================
def train_and_validate(X_train, y_train, X_test, num_epochs=50, batch_size=64, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_probs = np.zeros((len(X_test), 5)) if len(X_test) > 0 else None

    if test_probs is not None:
        test_dataset = VibrationDataset(X_test, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"--- Starting Fold {fold + 1}/5 ---")

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
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    preds = torch.argmax(model(inputs.to(device)), dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(labels.numpy())

            macro_f1 = f1_score(val_targets, val_preds, average='macro')

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_model_weights = copy.deepcopy(model.state_dict())

        print(f"Fold {fold + 1} Best Macro-F1: {macro_f1:.4f}")

        '''# 将每一折最好的模型权重保存到本地
        save_path = f'best_model_fold_{fold + 1}.pth'
        torch.save(best_model_weights, save_path)
        print(f"模型权重已保存至: {save_path}")'''

        # 推理测试集
        if test_probs is not None:
            model.load_state_dict(best_model_weights)
            model.eval()
            fold_probs = []
            with torch.no_grad():
                for inputs in test_loader:
                    outputs = torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy()
                    fold_probs.extend(outputs)
            test_probs += np.array(fold_probs) / kf.n_splits

    return test_probs


if __name__ == "__main__":
    train_files = [
        'beng/train_424000.csv',
        'gundongti/train_424000.csv',
        'neihuan/train_424000.csv',
        'waihuan/train_424000.csv',
        'zhengchang/train_424000.csv'
    ]
    test_file = 'test_106000.csv'

    all_train_data, all_train_labels = [], []
    for label, file_path in enumerate(train_files):
        df = pd.read_csv(file_path)
        signal_data = df.iloc[0:424, 0:1000].values
        all_train_data.append(signal_data)
        all_train_labels.append(np.full(len(signal_data), label))

    X_train_raw = np.vstack(all_train_data)
    y_train = np.concatenate(all_train_labels)

    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        test_ids = test_df.iloc[:, 0].values
        X_test_raw = test_df.iloc[:, 1:1001].values
    else:
        X_test_raw = None

    mean = np.mean(X_train_raw, axis=1, keepdims=True)
    std = np.std(X_train_raw, axis=1, keepdims=True) + 1e-8
    X_train = (X_train_raw - mean) / std

    if X_test_raw is not None:
        X_test = (X_test_raw - np.mean(X_test_raw, axis=1, keepdims=True)) / (
                    np.std(X_test_raw, axis=1, keepdims=True) + 1e-8)
    else:
        X_test = np.array([])

    test_ensemble_probs = train_and_validate(X_train, y_train, X_test, num_epochs=50, batch_size=64)

    '''if test_ensemble_probs is not None:
        submission = pd.DataFrame({'id': test_ids, 'label': np.argmax(test_ensemble_probs, axis=1)})
        submission.to_csv('fault_detection_results.csv', index=False)
        print("所有折训练完毕，预测结果已保存至 'fault_detection_results.csv'。")'''
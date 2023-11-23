"""
@Description :   CNN
@Author      :   Xubo Luo 
@Time        :   2023/11/23 17:17:11
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from utils import load_data

# 读取数据, 划分训练集和测试集
X_train_scaled, X_test_scaled, y_train, y_test = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 将数据转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # 添加一维表示通道
y_train_tensor = torch.FloatTensor(y_train.values)

X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test.values)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义一维CNN模型
# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 4, 50)
#         self.fc2 = nn.Linear(50, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return torch.sigmoid(x)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer_1 = nn.Linear(X_train_scaled.shape[1],64)
        self.layer_2 = nn.Linear(40,64)
        self.layer_out = nn.Linear(64,1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(inputs))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

# 初始化模型、损失函数和优化器
model = CNNModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# 在测试集上进行预测
with torch.no_grad():
    model.eval()
    y_pred = model(X_test_tensor).squeeze().numpy()
    y_pred_binary = np.round(y_pred)

# 评估模型性能
accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_binary)
report = classification_report(y_test_tensor.numpy(), y_pred_binary)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

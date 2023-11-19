"""
@Description :   CNN手写数字识别
@Author      :   Xubo Luo 
@Time        :   2023/11/17 10:56:45
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from utils import load_data
from sklearn.preprocessing import StandardScaler

# 从.mat文件加载数据
train_features, train_labels, test_features, test_labels = load_data('mnist_all.mat')

# 数据预处理
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# 转换为PyTorch Tensor
train_features = torch.FloatTensor(train_features).unsqueeze(1).reshape(-1,1,28,28)
train_labels = torch.LongTensor(train_labels)
test_features = torch.FloatTensor(test_features).unsqueeze(1).reshape(-1,1,28,28)
test_labels = torch.LongTensor(test_labels)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
is_eval = False

if not is_eval:
    # 训练模型
    train_losses = []
    num_epochs = 5
    for epoch in trange(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                train_losses.append(loss.item())

    # 保存模型
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')

    # 绘制损失函数曲线
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if is_eval:
    # 加载模型
    model.load_state_dict(torch.load('mnist_cnn_model.pth'))

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
"""
@Description :   确定最佳PCA维度
@Author      :   Xubo Luo 
@Time        :   2023/11/27 21:04:11
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import load_data

# 读取数据
X_train, y_train, X_test, y_test = load_data("mnist_all.mat")
print(X_train.shape)
# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 创建PCA模型
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 查看各主成分的方差贡献率
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# 绘制方差贡献率的累积图
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('Cumulative Variance Ratio vs. Number of Principal Components')
plt.show()

"""
@Description :   线性SVM手写数字识别
@Author      :   Xubo Luo 
@Time        :   2023/11/17 10:27:57
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.io import loadmat

# 加载手写数字数据集
dataset = loadmat('mnist_all.mat')

# 加载测试集和训练集
X_test = dataset['test0']
y_test = np.zeros(X_test.shape[0])
for i in range(1, 10):
    X_test = np.vstack((X_test, dataset['test' + str(i)]))
    y_test = np.hstack((y_test, np.full(dataset['test' + str(i)].shape[0], i)))

X_train = dataset['train0']
y_train = np.zeros(X_train.shape[0])
for i in range(1, 10):
    X_train = np.vstack((X_train, dataset['train' + str(i)]))
    y_train = np.hstack((y_train, np.full(dataset['train' + str(i)].shape[0], i)))


# 创建SVM模型
svm_model = SVC(kernel='linear')

# 训练模型
svm_model.fit(X_train, y_train)

# 预测测试集
y_pred = svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

"""
@Description :   Boosting手写数字识别
@Author      :   Xubo Luo 
@Time        :   2023/11/17 10:51:47
"""

# 导入所需的库
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import numpy as np

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

# 使用决策树作为基分类器
base_classifier = DecisionTreeClassifier(max_depth=1)

# 设置Boosting的迭代次数
n_estimators = 50

# 创建AdaBoost分类器
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=n_estimators, random_state=42)

# 训练模型
adaboost_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = adaboost_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

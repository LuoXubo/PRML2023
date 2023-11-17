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
import matplotlib.pyplot as plt
import _pickle as pickle

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

# 存储每次迭代后的训练集和测试集准确率
train_accuracies = []
test_accuracies = []

# 创建AdaBoost分类器
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=n_estimators, random_state=42)
is_eval = False

if not is_eval:
    # 训练模型
    for i, y_pred_train in enumerate(adaboost_classifier.staged_predict(X_train)):
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_accuracies.append(train_accuracy)

        y_pred_test = adaboost_classifier.estimators_[i].predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_accuracies.append(test_accuracy)

        print(f"Iteration {i + 1}: Training Accuracy = {train_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")
    # 训练模型
    adaboost_classifier.fit(X_train, y_train)

    # 保存模型
    with open('boosting.pkl', 'wb') as f:
        pickle.dump(adaboost_classifier, f)

    # 绘制每次迭代的准确率曲线
    plt.plot(range(1, n_estimators + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, n_estimators + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# 加载模型
if is_eval:
    with open('boosting.pkl', 'rb') as f:
        adaboost_classifier = pickle.load(f)

# 在测试集上进行预测
y_pred = adaboost_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

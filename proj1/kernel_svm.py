"""
@Description :   Kernel SVM手写数字识别
@Author      :   Xubo Luo 
@Time        :   2023/11/19 13:14:52
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import _pickle as pickle
from utils import load_data


# 初始化SVM分类器，选择RBF（径向基函数）内核
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')
print('Init SVM model')
is_eval = True

# 加载手写数字数据集
X_train, y_train, X_test, y_test = load_data('mnist_all.mat')

# 训练模型
if not is_eval:
    n_split = 5
    accuracy_scores = np.array([])
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    with tqdm(total=n_split) as pbar:
        pbar.set_description('Train SVM model')
        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            svm_classifier.fit(X_train_fold, y_train_fold)
            X_val_fold_pred = svm_classifier.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, X_val_fold_pred)
            accuracy_scores = np.append(accuracy_scores, accuracy)
            print(f"Fold {fold} Accuracy: {accuracy * 100:.2f}%")
            pbar.update(1)
    # 保存模型
    with open('svm_classifier.pkl', 'wb') as f:
        pickle.dump(svm_classifier, f)
    x_axis = np.arange(1, n_split + 1)
    plt.plot(x_axis, accuracy_scores, linewidth=1, label='accuracy')
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.show()

# 加载模型
if is_eval:
    with open('svm_classifier.pkl', 'rb') as f:
        svm_classifier = pickle.load(f)

# 预测测试集
y_pred = svm_classifier.predict(X_test)
print('Predict test set')

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

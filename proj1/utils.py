"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2023/11/19 14:02:32
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
from sklearn.decomposition import PCA

def load_data(filename):
    """
    加载数据集
    """
    dataset = loadmat(filename)
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
    return X_train, y_train, X_test, y_test

def pca_fit(X_input, n_components):
    
    pca = PCA(n_components=n_components)  # 指定降维后的维度
    X_pca = pca.fit_transform(X_input)
    return X_pca
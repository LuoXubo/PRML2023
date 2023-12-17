"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2023/12/12 21:17:55
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import argparse
from scipy.optimize import linear_sum_assignment

from utils import *

allLabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__' :
    parser = argparse.ArgumentParser ()
    parser.add_argument('--eval', type = int, default = 0, help = 'evaluate the model')
    parser.add_argument('--file_path', type = str, default = '../../cifar-10-python/', help = 'path of cifar-10-python')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size for training')
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs for training')
    args = parser.parse_args ()
    file_path = args.file_path
    batch_size = args.batch_size
    epochs = args.epochs


    ''' 读取 CIFAR-10 数据集 '''
    train_set, test_set = DataLoad(batch_size=batch_size, num_workers=2, path=file_path)
    print('number of batches in train set: %d'%len(train_set))
    print('number of batches in test set: %d'%len(test_set))
    # print (len (train_set), len (test_set))
    print (next (iter (train_set))[0].shape)
    
    criterion = nn.CrossEntropyLoss ()
    # knn = KNeighborsClassifier(n_neighbors=5)
    # model = GaussianNB()
    # model = LinearRegression()
    model = LogisticRegression(max_iter=3000)
    test_acc = []
    for epoch in range(1, epochs + 1) :
        n, acc_sum = 0, 0
        for X, y in train_set :
            X = X.reshape(X.shape[0], -1)
            model.fit(X, y)
        #     y_pred = knn.predict(X)
        #     y_pred = torch.from_numpy(y_pred).type(torch.FloatTensor)
        #     print(y_pred)
        #     print(y)
        #     loss = criterion(y_pred, y.type(torch.FloatTensor)).sum()
        #     n += y.shape[0]
        #     acc_sum += (y_pred == y).sum().item()
        #     losses.append(loss.item() / n)
        # train_acc.append(acc_sum / n)

        acc_ratio, n = 0.0, 0
        ncnt = [0]*10
        for X, y in test_set :
            X = X.reshape(X.shape[0], -1)
            y_pred = model.predict(X)
            y = y.numpy()
            acc_ratio += (y_pred == y).sum().item()
            # acc_ratio += (y_pred == y.numpy()).float().sum().item()
            n += y.shape[0]
            for i in range(y.shape[0]):
                if y_pred[i] != y[i]:
                    ncnt[y[i]] += 1
        acc_ratio /= n
        test_acc.append(acc_ratio)
        
        print('[%d/%d] , test acc : %.3f'
          % (epoch, epochs, test_acc[-1]))
    
    DrawBarChart (allLabels, ncnt)
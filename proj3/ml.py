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

from utils import cluster_acc, knn_gnb_lr_lsr, get_imgdata, DataLoad

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
    print (len (train_set), len (test_set))
    print (next (iter (train_set))[0].shape)
    
    criterion = nn.CrossEntropyLoss ()
    knn = KNeighborsClassifier(n_neighbors=5)
    for epoch in range(1, epochs + 1) :
        n, acc_sum = 0, 0
        for X, y in train_set :
            X = X.reshape(X.shape[0], -1)
            knn.fit(X, y)
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
            y_pred = knn.predict(X)
            print(type(y))
            print(y_pred== y )
            acc_ratio += (y_pred == y).float().sum().item()
            n += y.shape[0]
            for i in range(y.shape[0]):
                if pred[i] == y[i]:
                    ncnt[y[i]] += 1
        acc_ratio /= n
        test_acc.append(acc_ratio)
        cnt = ncnt

        # 打印训练日志
        # print('[%d/%d] loss : %.4f, train acc : %.3f, test acc : %.3f'
        #   % (epoch, epochs, losses[-1], train_acc[-1], test_acc[-1]))
        
        print('[%d/%d] , test acc : %.3f'
          % (epoch, epochs, test_acc[-1]))
        
    # knn_acc = []
    # gnb_acc = []
    # lr_acc = []
    # lsr_acc = []
    # for i in range(n_test):
    #     t1,t2,t3,t4= \
    #         knn_gnb_lr_lsr(X_[i], Y_[i])
    #     knn_acc.append(t1)
    #     gnb_acc.append(t2)
    #     lr_acc.append(t3)
    #     lsr_acc.append(t4)
    # # 使用pandas输出
    # title1 = []
    # for i in range(n_test):
    #     t = 'data_bath'+str(i+1)
    #     title1.append(t)
    # title2 = ["KNN     ", "Naive Bayes","linear regression","Logistic regression"]
    # data = pd.DataFrame([knn_acc,gnb_acc,lr_acc,lsr_acc],index=title2,columns=title1)
    # print(data)
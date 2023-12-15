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
import argparse

from scipy.optimize import linear_sum_assignment

from utils import cluster_acc, knn_gnb_lr_lsr, get_imgdata, DataLoad

if __name__ == '__main__' :
    parser = argparse.ArgumentParser ()
    parser.add_argument('--file_path', type = str, default = '../../cifar-10-python/cifar-10-batches-py/batches.meta', help = 'path of cifar-10-batches-py')
    parser.add_argument('--data_path', type = str, default = '../../cifar-10-python/cifar-10-batches-py/data_batch_', help = 'path of cifar-10-batches-py')
    parser.add_argument('--n_test', type = int, default = 5, help = 'number of test data')
    parser.add_argument('--re_size', type = int, default = 32, help = 'resize the image')

    args = parser.parse_args ()
    file = args.file_path
    sfile = args.data_path
    n_test = args.n_test
    re_size = args.re_size

    ''' 读取 CIFAR-10 数据集 '''
    train_set, test_set = DataLoad(batch_size=batch_size, num_workers=2, path=file)
    print (len (train_set), len (test_set))
    print (next (iter (train_set))[0].shape)

    X,Y,Z = get_imgdata(file,sfile,re_size,n_test)
    X = X.reshape(n_test*10000, -1)
    X_ = []
    Y_ = []
    for i in range(n_test):
        X_.append(X[i*10000:(i+1)*10000])
        Y_.append(Y[i*10000:(i+1)*10000])
    X_ = np.array(X_)
    Y_ = np.array(Y_)
    print(X_.shape)
    print(Y_.shape)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    for epoch in range(1, epochs + 1) :
        n, acc_sum = 0, 0
        for X, y in train_set :
           knn.fit(X, y)
           y_pred = knn.predict(X)
           loss = criterion(y_pred, y).sum()
           n += y.shape[0]
           acc_sum += (y_pred == y).sum().item()
        losses.append(loss.item() / n)
        train_acc.append(acc_sum / n)

        acc_ratio, n = 0.0, 0
        ncnt = [0]*10
        for X, y in test_set :
            y_pred = knn.predict(X)
            acc_ratio += (y_pred == y).float().sum().item()
            n += y.shape[0]
            for i in range(y.shape[0]):
                if pred[i] == y[i]:
                    ncnt[y[i]] += 1
        acc_ratio /= n
        test_acc.append(acc_ratio)
        cnt = ncnt

        # 打印训练日志
        print('[%d/%d] loss : %.4f, train acc : %.3f, test acc : %.3f'
          % (epoch, epochs, losses[-1], train_acc[-1], test_acc[-1]))
        
    knn_acc = []
    gnb_acc = []
    lr_acc = []
    lsr_acc = []
    for i in range(n_test):
        t1,t2,t3,t4= \
            knn_gnb_lr_lsr(X_[i], Y_[i])
        knn_acc.append(t1)
        gnb_acc.append(t2)
        lr_acc.append(t3)
        lsr_acc.append(t4)
    # 使用pandas输出
    title1 = []
    for i in range(n_test):
        t = 'data_bath'+str(i+1)
        title1.append(t)
    title2 = ["KNN     ", "Naive Bayes","linear regression","Logistic regression"]
    data = pd.DataFrame([knn_acc,gnb_acc,lr_acc,lsr_acc],index=title2,columns=title1)
    print(data)
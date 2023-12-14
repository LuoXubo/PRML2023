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
from scipy.optimize import linear_sum_assignment

from utils import cluster_acc, knn_gnb_lr_lsr, get_imgdata

file = '../../cifar-10-python/cifar-10-batches-py/batches.meta'
sfile = '../../cifar-10-python/cifar-10-batches-py/data_batch_'

n_test = 5
re_size = 8

X,Y,Z = get_imgdata(file,sfile,re_size,n_test)
X = X.reshape(n_test*10000, -1)
X_ = []
Y_ = []
for i in range(n_test):
    X_.append(X[i*10000:(i+1)*10000])
    Y_.append(Y[i*10000:(i+1)*10000])
X_ = np.array(X_)
Y_ = np.array(Y_)

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
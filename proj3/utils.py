"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2023/12/08 09:00:45
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib_inline
from pylab import mpl

import random
import numpy as np

transform = transforms.Compose ([
    transforms.ToTensor (),
    transforms.RandomHorizontalFlip (),
    transforms.Normalize ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")

def DataLoad (batch_size = 64, num_workers = 2) :
  # 数据的读取
  cifar10_train = torchvision.datasets.CIFAR10 (
    root = '../../cifar-10-python', 
    train = True, download = False, 
    transform = transform
  )
  cifar10_test = torchvision.datasets.CIFAR10 (
    root = '../../cifar-10-python', 
    train = False, download = False, 
    transform = transform
  )

  # 数据打包成批次
  train_set = Data.DataLoader (
    dataset = cifar10_train,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
  )
  test_set = Data.DataLoader (
    dataset = cifar10_test,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
  )
  
  return train_set, test_set

def DrawDoubleLineChart (losses, train_acc, test_acc, figsize = (10, 5)) :
  matplotlib_inline.backend_inline.set_matplotlib_formats ('svg')
  plt.rcParams['figure.figsize'] = figsize
  n = len (losses); x = range (1, n + 1)
  plt.subplot (1, 2, 1)
  plt.grid ()
  plt.plot (x, losses, marker = 'o')
  plt.xlabel ("epoch")
  plt.ylabel ("losses")
  plt.subplot (1, 2, 2)
  plt.grid ()
  plt.plot (x, train_acc, marker = 'o', color = "black", label = "train_acc")
  plt.plot (x, test_acc, marker = 'o', color = "blue", label = "test_acc")
  plt.xlabel ("epoch")
  plt.ylabel ("accuracies")
  plt.show ()
  
''' 绘制不同种类图像的分类错误率 '''
def DrawBarChart (categories, cnt, figsize = (5, 5)) :
  matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
  plt.rcParams['figure.figsize'] = figsize
  _, ax = plt.subplots()
  b = ax.barh (range (len (categories)), cnt, color = '#6699CC')
  
  # 为横向水平的柱图右侧添加数据标签。
  for rect in b:
      w = rect.get_width ()
      ax.text (w, rect.get_y () + rect.get_height () /2, '%d' %
              int (w), ha = 'left', va = 'center')
  
  # 设置Y轴纵坐标上的刻度线标签。
  ax.set_yticks (range (len(categories)))
  ax.set_yticklabels (categories)
  
  # 不要X横坐标上的label标签。
  plt.xticks (())
  plt.title ('The number of incorrect classification \n in different categories', 
             loc = 'center', fontsize = '15', color = 'black')
  plt.show ()
  
''' 评估函数, 用于计算模型在测试集上的准确率 '''

def Evaluate (net, Dataset) :
  acc_ratio, n = 0.0, 0
  ncnt = [0] * 10
  with torch.no_grad () :
    for X, y in Dataset :
      net.eval ()
      out = net (X.to (device))
      _, pred = torch.max (out, 1)
      acc_ratio += (pred == y.to (device)).float ().sum ().cpu ().item ()
      net.train ()
      n += y.shape[0]
      for i in range (y.shape[0]) :
        if pred[i] != y[i] : ncnt[y[i]] += 1
  acc_ratio /= n
  return acc_ratio, ncnt

''' 训练过程的实现 '''

def Trainer (train_set, test_set, net, epochs, criterion, optimizer) :
  losses, train_acc, test_acc = [], [], []
  cnt = [0] * 10
  for epoch in range (1, epochs + 1) :
    n, acc_sum = 0, 0
    for X, y in train_set :
      X = X.to (device)
      out = net (X)
      loss = criterion (out, y.to (device)).sum ()
      optimizer.zero_grad ()
      loss.backward ()
      optimizer.step ()
      n += y.shape[0]
      acc_sum += (out.argmax (dim = 1) == y.to (device)).sum ().item ()
    losses.append (loss.item ())
    train_acc.append (acc_sum / n)
    nacc, ncnt = Evaluate (net, test_set)
    test_acc.append (nacc)
    cnt = ncnt
    # 打印训练日志
    print('[%d/%d] loss : %.4f, train acc : %.3f, test acc : %.3f'
          % (epoch, epochs, losses[-1], train_acc[-1], test_acc[-1]))
  torch.save (net.state_dict (), 'resnet18.pth')
  return losses, train_acc, test_acc, cnt

# functions used in machine learning methods
def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def get_imgdata(file,sfile,re_size=16,n=5):
    import pickle
    import numpy as np
    from skimage.transform import resize

    def unpickle(file):
        with open(file, 'rb') as f:
            cifar_dict = pickle.load(f, encoding='latin1')
        return cifar_dict
    # 定义用来存放图像数据 图像标签 图像名称list  最后返回的cifar_image cifar_label即是图像cifar-10 对应的数据和标签
    tem_cifar_image = []
    tem_cifar_label = []
    tem_cifar_image_name = []
    for i in range(1, n+1):
        # 存放是你的文件对应的目录
        cifar_file = sfile + str(i)
        cifar = unpickle(cifar_file)
        cifar_label = cifar['labels']
        cifar_image = cifar['data']
        cifar_image_name = cifar['filenames']
        # 使用transpose()函数是因为cifar存放的是图像标准是 通道数 高 宽 所以要修改成  高 宽 通道数
        cifar_image = cifar_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        cifar_image = np.asarray([resize(x_img, [re_size, re_size]) for x_img in cifar_image])
        cifar_label = np.array(cifar_label)
        cifar_image_name = np.array(cifar_image_name)
        tem_cifar_image.append(cifar_image)
        tem_cifar_label.append(cifar_label)
        tem_cifar_image_name.append(cifar_image_name)
    cifar_image = np.concatenate(tem_cifar_image)
    cifar_label = np.concatenate(tem_cifar_label)
    cifar_image_name = np.concatenate(tem_cifar_image_name)
    return cifar_image, cifar_label, cifar_image_name

## 朴素贝叶斯与KNN分类
def knn_gnb_lr_lsr(X, labels, title_knn="CIFAR.KNN",\
                title_gnb="CIFAR.GNB",title_lr="CIFAR.LR",\
                title_lsr="CIFAR.LSR",n = 3):
    # 划分训练集和测试集
    x_train, x_test, labels_train, labels_test =\
        train_test_split(X, labels, test_size=0.2, random_state=22)

    # 使用KNN进行分类
    knn = KNeighborsClassifier()
    knn.fit(x_train, labels_train)
    label_sample = knn.predict(x_test)
    knn_acc=cluster_acc(labels_test, label_sample)
    print(title_knn,"=",knn_acc)

    # 使用高斯朴素贝叶斯进行分类
    gnb = GaussianNB()  # 使用默认配置初始化朴素贝叶斯
    gnb.fit(x_train, labels_train)  # 训练模型
    label_sample = gnb.predict(x_test)
    gnb_acc = cluster_acc(labels_test, label_sample)
    print(title_gnb,"=", gnb_acc)

    # 线性回归
    lr = LinearRegression()
    lr.fit(x_train, labels_train)
    label_sample = lr.predict(x_test)
    label_sample = np.round(label_sample)
    label_sample=label_sample.astype(np.int64)
    lr_acc = cluster_acc(labels_test, label_sample)
    print(title_lr, "=", lr_acc)

    #Logistic regression 需要事先进行标准化
    #创建一对多的逻辑回归对象
    # 标准化特征
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X,labels)
    # 划分训练集和测试集
    x_train, x_test, labels_train, labels_test = \
        train_test_split(X_, labels, test_size=0.2)
    log_reg = LogisticRegression(max_iter=3000)#multinomial
    #训练模型
    log_reg.fit(x_train, labels_train)
    label_sample = log_reg.predict(x_test)
    lsr_acc = cluster_acc(labels_test, label_sample)
    print(title_lsr, "=", lsr_acc)
    print('\n')

    return round(knn_acc,n),round(gnb_acc,n),round(lr_acc,n),round(lsr_acc,n)

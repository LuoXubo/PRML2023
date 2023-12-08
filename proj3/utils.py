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

def DataLoad (batch_size = 64, num_workers = 2) :
  # 数据的读取
  cifar10_train = torchvision.datasets.CIFAR10 (
    root = '/kaggle/input/cifar10-python', 
    train = True, download = False, 
    transform = transform
  )
  cifar10_test = torchvision.datasets.CIFAR10 (
    root = '/kaggle/input/cifar10-python', 
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
  return losses, train_acc, test_acc, cnt

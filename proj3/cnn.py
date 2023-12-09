"""
@Description :   ResNet for CIFAR-10
@Author      :   Xubo Luo 
@Time        :   2023/12/08 08:57:30
"""

''' 导入实验所必需的库 '''
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

from utils import *
from resnet import ResNet18, ResBlock

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
is_eval = False


device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")

''' 读取 CIFAR-10 数据集 '''
batch_size = 64
train_set, test_set = DataLoad (batch_size)
print (len (train_set), len (test_set))
print (next (iter (train_set))[0].shape)

''' 残差网络 18 模型的构建 '''
net18 = ResNet18 (ResBlock).to (device)
print (net18)

if not is_eval :
    ''' 训练模型 '''
    allLabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    epochs = 50
    optimizer = optim.Adam (net18.parameters (), lr = 0.0002)
    criterion = nn.CrossEntropyLoss ()
    losses, train_acc, test_acc, cnt = Trainer (train_set, test_set, net18, epochs, criterion, optimizer)
    DrawDoubleLineChart (losses, train_acc, test_acc)
    DrawBarChart (allLabels, cnt)

else:
    ''' 加载已经训练好的模型 '''
    net18.load_state_dict (torch.load ('./net18.pth'))
    net18.eval ()

    ''' 测试模型 '''
    correct = 0; total = 0
    with torch.no_grad () :
        for data in test_set :
            images, labels = data
            images, labels = images.to (device), labels.to (device)
            outputs = net18 (images)
            _, predicted = torch.max (outputs.data, 1)
            total += labels.size (0)
            correct += (predicted == labels).sum ().item ()
    print ('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    ''' 每个类别的准确率 '''
    class_correct = list (0. for i in range (10))
    class_total = list (0. for i in range (10))
    with torch.no_grad () :
        for data in test_set :
            images, labels = data
            images, labels = images.to (device), labels.to (device)
            outputs = net18 (images)
            _, predicted = torch.max (outputs, 1)
            c = (predicted == labels).squeeze ()
            for i in range (4) :
                label = labels[i]
                class_correct[label] += c[i].item ()
                class_total[label] += 1
    for i in range (10) :
        print ('Accuracy of %5s : %2d %%' % (allLabels[i], 100 * class_correct[i] / class_total[i]))

    ''' 混淆矩阵 '''
    confusion_matrix = np.zeros ((10, 10))
    with torch.no_grad () :
        for data in test_set :
            images, labels = data
            images, labels = images.to (device), labels.to (device)
            outputs = net18 (images)
            _, predicted = torch.max (outputs, 1)
            for i in range (4) :
                confusion_matrix[labels[i]][predicted[i]] += 1
    DrawConfusionMatrix (confusion_matrix, allLabels)
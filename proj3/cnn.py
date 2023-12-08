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
from resnet import ResNet18

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

transform = transforms.Compose ([
    transforms.ToTensor (),
    transforms.RandomHorizontalFlip (),
    transforms.Normalize ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")

''' 读取 CIFAR-10 数据集 '''
batch_size = 64
train_set, test_set = DataLoad (batch_size)
print (len (train_set), len (test_set))
print (next (iter (train_set))[0].shape)

''' 残差网络 18 模型的构建 '''
net18 = ResNet18 (ResBlock).to (device)
print (net18)

''' 训练模型 '''
allLabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']
epochs = 50
optimizer = optim.Adam (net18.parameters (), lr = 0.0002)
criterion = nn.CrossEntropyLoss ()
losses, train_acc, test_acc, cnt = Trainer (train_set, test_set, net18, epochs, criterion, optimizer)
DrawDoubleLineChart (losses, train_acc, test_acc)
DrawBarChart (allLabels, cnt)
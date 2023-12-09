"""
@Description :   ResNet structure
@Author      :   Xubo Luo 
@Time        :   2023/12/08 09:01:24
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

device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")


class ResBlock (nn.Module) :
  '''
    第一个残差块使输入输出维度缩小一半, 残差时需要调整原先输入的维度大小 (同样利用卷积操作)
    第二个残差块输入输出维度相同, 归一化后直接相加即可
  '''
  
  def __init__ (self, in_channels, out_channels) :
    super (ResBlock, self).__init__ ()

    self.block1 = nn.Sequential (
      # (b, c_in, x, x) -> (b, c_out, x / 2, x / 2)
      nn.Conv2d (in_channels, out_channels, 4, 2, 1, bias = False),
      nn.Dropout (0.45),
      nn.BatchNorm2d (out_channels),
      nn.ReLU (inplace = True),
      # (b, c_out, x, x) -> (b, c_out, x, x)
      nn.Conv2d (out_channels, out_channels, 3, 1, 1, bias = False),
      nn.Dropout (0.45),
      nn.BatchNorm2d (out_channels)
    )

    self.cut = nn.Sequential (
      # (b, c_in, x, x) -> (b, c_out, x / 2, x / 2)
      nn.Conv2d (in_channels, out_channels, 2, 2, 0, bias = False),
      nn.Dropout (0.45),
      nn.BatchNorm2d (out_channels)
    )

    self.block2 = nn.Sequential (
      # (b, c_out, x, x) -> (b, c_out, x, x)
      nn.Conv2d (out_channels, out_channels, 3, 1, 1, bias = False),
      nn.Dropout (0.45),
      nn.BatchNorm2d (out_channels),
      nn.ReLU (inplace = True),
      nn.Conv2d (out_channels, out_channels, 3, 1, 1, bias = False),
      nn.Dropout (0.45),
      nn.BatchNorm2d (out_channels)
    )

  def forward (self, X) :
    out = self.block1 (X) + self.cut (X)
    out = out + self.block2 (out)
    out = F.relu (out)
    return out
  
class ResNet18 (nn.Module) :
  '''
    input is (batch_size, 3, 32, 32)
    一个卷积层 + 四个残差块 + 平均池化 + 全连接层
    一个卷积层: 3 通道扩充为 64 通道, 图像大小不变
    四个残差块: 每个残差块过后图像长宽缩小为原来的一半, 32 -> 32 -> 16 -> 8 -> 4
                                    通道数变化: 64 -> 64 -> 128 -> 256 -> 512
    平均池化: 4 * 4 大小, 平均池化操作后实际上每个通道上都是一个 1 * 1 大小的图
    全连接层: 512 -> 10 (class_num)
  '''
  def __init__ (self, ResBlock) :
    super (ResNet18, self).__init__ ()
    
    self.conv = nn.Sequential (
      nn.Conv2d (3, 64, 3, 1, 1, bias = False),
      nn.Dropout (0.45),
      nn.BatchNorm2d (64),
      nn.ReLU ()
    )

    self.layer1 = ResBlock (64, 64)
    self.layer2 = ResBlock (64, 128)
    self.layer3 = ResBlock (128, 256)
    self.layer4 = ResBlock (256, 512)

    self.pool = nn.AvgPool2d (2)
    self.fc = nn.Linear (512, 10)

  def forward (self, X) :
    out = self.conv (X)
    out = self.layer1 (out)
    out = self.layer2 (out)
    out = self.layer3 (out)
    out = self.layer4 (out)
    out = self.pool (out)
    out = out.view (out.shape[0], -1)
    out = self.fc (out)
    return out

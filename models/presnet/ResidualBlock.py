#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:30:53 2020

@author: emma
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class ResidualBlock(nn.Module):

    def __init__(self, channels1,channels2,res_stride=1):
        super(ResidualBlock, self).__init__()
        self.inplanes=channels1
        # Exercise 2.1.1 construct the block without shortcut
        self.conv1 = nn.Conv2d(channels1, channels2, kernel_size=3, 
                               stride=res_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels2)
        self.conv2 = nn.Conv2d(channels2, channels2, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels2)

        if res_stride != 1 or channels2 != channels1:
        # Exercise 2.1.3 the shortcut; create option for resizing input 
            self.shortcut=nn.Sequential(
                nn.Conv2d(channels1, channels2, kernel_size=1, 
                          stride=res_stride, bias=False),
                nn.BatchNorm2d(channels2)
            )
        else:
            self.shortcut=nn.Sequential()
            

    def forward(self, x):
        
        # forward pass: Conv2d > BatchNorm2d > ReLU > 
        #Conv2D >  BatchNorm2d > ADD > ReLU
        out=self.conv1(x)
        out=self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # THIS IS WHERE WE ADD THE INPUT
        #print('input shape',x.shape,self.inplanes)
        out += self.shortcut(x)
       # print('res block output shape',  out.shape)
        # final ReLu
        out = F.relu(out)

        return out
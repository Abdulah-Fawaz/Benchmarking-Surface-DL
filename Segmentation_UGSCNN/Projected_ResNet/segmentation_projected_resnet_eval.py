#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:47:14 2021

@author: logan
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

import math
import argparse

import numpy as np
import pickle, gzip
import logging
import shutil

from ResidualBlock import ResidualBlock
from model import UNet

import torch
import torch.nn.functional as F
import torch.optim as optim

from MyDataLoader import My_Projected_dHCP_Data_Segmentation, My_Projected_dHCP_Data_Segmentation_Test
from torch import nn
import matplotlib.pyplot as plt

def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.
    epsilon = 10e-8

    # have to use contiguous since they may from a torch.view op
    iflat = pred.view(-1).contiguous()
    tflat = target.view(-1).contiguous()
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    dice = dice.mean(dim=0)
    dice = torch.clamp(dice, 0, 1.0)

    return  dice

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('/home/lw19/Desktop/final_benchmarking_models/best_Projected_UNet_Native_warp_no_rot')
model.to(device)

test_set = np.load('/data/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_test_TEA.npy', allow_pickle=True)
test_loader = My_Projected_dHCP_Data_Segmentation(test_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                     normalisation='std', parity_choice='both', output_as_torch=True, projected=True)
rot_test_loader = My_Projected_dHCP_Data_Segmentation(test_set, number_of_warps = 0, rotations=True, smoothing = False, 
                                     normalisation='std', parity_choice='both', output_as_torch=True, projected=True)
MyTestLoader = DataLoader(test_loader, batch_size=1 ,shuffle=False, num_workers=2)
MyRotTestLoader = DataLoader(rot_test_loader, batch_size=1 ,shuffle=False, num_workers=2)

test_loss = []
model.eval()
for i, batch in enumerate(MyTestLoader):
    test_images = batch['image']
    test_images = test_images.to(device)
    test_label = batch['label'].to(device)
    estimates = model(test_images)
    loss = dice_coeff(estimates,test_label)
    print(loss)
    test_loss.append(loss.item())
print('loss', np.mean(test_loss))
print('SD loss', np.std(test_loss))

test_loss = []
for i, batch in enumerate(MyRotTestLoader):
    test_images = batch['image']
    

    test_images = test_images.to(device)
    test_labels = batch['label'].to(device)

    test_labels = test_labels.unsqueeze(1)

    estimates = model(test_images)
    loss = dice_coeff(estimates,test_labels)

    test_loss.append(loss.item())
print('loss', np.mean(test_loss))
print('SD loss', np.std(test_loss))

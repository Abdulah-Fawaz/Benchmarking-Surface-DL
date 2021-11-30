#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:31:47 2021

@author: logan
"""
import math
import argparse
import sys, os; sys.path.append("/data/Data/ugscnn1/meshcnn")
import numpy as np
import pickle, gzip
import logging
import shutil
from utils import sparse2tensor, spmatmul, My_Projected_dHCP_Data, My_Projected_dHCP_Data_Test
from ops import MeshConv
from model import Model, Model_confound
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device is ', device)

criterion = nn.MSELoss()
batch_size = 8 # batch size can be any number
learning_rate = 1e-4
feat=32
mesh_folder='/media/logan/Storage/Data/ugscnn1/mesh_files'
numberOfEpochs = 20000

print("Batch Size is ", batch_size)
print("Loss Function is ", criterion)
print("Learning Rate is ", learning_rate)
print("Total Number of Epochs is ", numberOfEpochs)

test_set = np.load('/data/Data/dHCP/Projected_ResNet/data/scan_age/test.npy', allow_pickle=True)
test_loader = My_Projected_dHCP_Data(test_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                     normalisation='std', parity_choice='both')
MyTestLoader = DataLoader(test_loader, batch_size=1 ,shuffle=False, num_workers=2)

rot_test_ds = My_Projected_dHCP_Data(test_set, number_of_warps = 0, rotations=True, smoothing = False, 
                           normalisation='std', parity_choice='both')
MyRotTestLoader =  torch.utils.data.DataLoader(rot_test_ds,1,   shuffle=False, num_workers=1)

model = torch.load('/home/lw19/Desktop/final_benchmarking_models/best_UG-SCNN_scan_age_Native_warp_no_rot')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.01)

test_outputs = []
test_labels = []
model.eval()
for i, batch in enumerate(MyTestLoader):
    test_images = batch['image']
    test_images = test_images.to(device)
    #metadata=batch['metadata'].to(device)
    test_label = batch['label'].to(device)
    test_output = model(test_images)#,metadata)
    test_outputs.append(test_output.item())
    test_labels.append(test_label.item())

print('unrotated average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
print('unrotated average absolute error STD', np.std(np.abs(np.array(test_outputs)-np.array(test_labels)))) 


test_outputs = []
test_labels = []
model.eval()
for i, batch in enumerate(MyRotTestLoader):
    test_images = batch['image']
    #metadata=batch['metadata'].to(device)
    test_images = test_images.to(device)
    test_label = batch['label'].to(device)
    test_output = model(test_images)#,metadata)
    test_outputs.append(test_output.item())
    test_labels.append(test_label.item())

print('rotated average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
print('unrotated average absolute error STD', np.std(np.abs(np.array(test_outputs)-np.array(test_labels)))) 



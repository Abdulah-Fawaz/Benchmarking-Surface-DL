#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:57:29 2021

@author: logan
"""
import sys, os; sys.path.append("/data/Data/ugscnn1/meshcnn")
from utils import sparse2tensor, spmatmul, My_Projected_dHCP_Data_Segmentation, My_Projected_dHCP_Data, My_Projected_dHCP_Data_Segmentation_Test
from ops import MeshConv
from model import SphericalUNet
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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


mesh_folder='/data/Data/ugscnn1/mesh_files'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load('/home/lw19/Desktop/final_benchmarking_models/best_SCNN-UG_Native_warp_rot')
model.to(device)


test_set = np.load('/data/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_test_TEA.npy', allow_pickle=True)
#test_set = np.load('/media/logan/Storage/Data/dHCP/Projected_ResNet/data/scan_age/test.npy', allow_pickle=True)
#test_set = np.load('/home/logan/Desktop/birth_age_confounded/test.npy',allow_pickle=True)

test_loader = My_Projected_dHCP_Data_Segmentation(test_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                     normalisation='std', parity_choice='left')

MyTestLoader = DataLoader(test_loader, batch_size=1 ,shuffle=False, num_workers=2)

rot_test_loader = My_Projected_dHCP_Data_Segmentation(test_set, number_of_warps = 0, rotations=True, smoothing = False, 
                                     normalisation='std', parity_choice='both')

MyRotTestLoader = DataLoader(rot_test_loader, batch_size=1 ,shuffle=False, num_workers=2)

criterion = nn.MSELoss()

test_loss = []
test_outputs = []
test_labels = []
scan_age = []
model.eval()
for i, batch in enumerate(MyTestLoader):
    test_images = batch['image']
    #metadata=batch['metadata'].to(device)
    test_images = test_images.to(device)
    test_label = batch['label'].to(device)

#    test_labels = test_labels.unsqueeze(1)

    estimates = model(test_images)
    #test_output = model(test_images,metadata).to(device).detach().cpu().numpy()
    
    loss = dice_coeff(estimates,test_label)
    #loss = criterion(test_output,test_label)
    print(loss)
    test_loss.append(loss.item())
    #test_outputs.append(test_output.item())
    #test_labels.append(test_label.item())
    #scan_age.append(metadata.item())
    #print(np.mean(test_loss))
#plt.scatter(x = scan_age, y = test_outputs)
print(np.mean(test_loss))
print(np.std(test_loss))

#print('unrotated average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 

#test_outputs = []
#test_labels = []
#test_loss = []
#scan_age = []
#model.eval()
#for i, batch in enumerate(MyRotTestLoader):
#    test_images = batch['image']
        
#    test_images = test_images.to(device)
#    test_label = batch['label'].to(device)
    #metadata=batch['metadata'].to(device)
    #test_labels = test_labels.unsqueeze(1)

#    estimates = model(test_images)
    #test_output = model(test_images,metadata).to(device).detach().cpu().numpy()
   # print(test_output)
#    loss = dice_coeff(estimates,test_label)
    #loss = criterion(test_output,test_label)
    
    #test_outputs.append(test_output.item())
    #test_labels.append(test_label.item())
    #scan_age.append(metadata.item())

    
    #print(loss)
#    test_loss.append(loss.item())
#print('rotated average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
#print(np.mean(test_loss))
#print(np.std(test_loss))
#plt.scatter(x = scan_age, y = test_outputs)

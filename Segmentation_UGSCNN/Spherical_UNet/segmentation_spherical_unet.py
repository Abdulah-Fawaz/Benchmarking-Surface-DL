#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:02:40 2020

@author: fa19
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
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Unet_40k

from Spherical_UNet_Dataloader import My_Projected_dHCP_Data_Segmentation, My_Projected_dHCP_Data_Segmentation_Test

from torch import nn
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device is ', device)

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
def main():
        
    batch_size = 1 # batch size can be any number
    learning_rate = 1e-3
    numberOfEpochs = 20000
    
    print("Batch Size is ", batch_size)
    print("Loss Function is 1- dice coefficient")
    print("Learning Rate is ", learning_rate)
    print("Total Number of Epochs is ", numberOfEpochs)
    
    train_set = np.load('../M-CRIB-S_train_TEA.npy', allow_pickle=True)
    val_set = np.load('../M-CRIB-S_val_TEA.npy', allow_pickle=True)
    test_set = np.load('..//M-CRIB-S_test_TEA.npy', allow_pickle=True)
        
    train_loader = My_Projected_dHCP_Data_Segmentation(train_set, number_of_warps = 99, rotations=True, smoothing = False, 
                                          normalisation='std', parity_choice='both', output_as_torch=True)
    val_loader = My_Projected_dHCP_Data_Segmentation(val_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                          normalisation='std', parity_choice='both', output_as_torch=True)
    test_loader = My_Projected_dHCP_Data_Segmentation(test_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                         normalisation='std', parity_choice='both', output_as_torch=True)
    
    rot_test_loader = My_Projected_dHCP_Data_Segmentation_Test(test_set, number_of_warps = 0, rotations=True, smoothing = False, 
                                         normalisation='std', parity_choice='both', output_as_torch=True)
    
    MyTrainLoader  = DataLoader(train_loader,batch_size=1, shuffle=False, num_workers=4)
    MyValLoader  = DataLoader(val_loader,batch_size=1, shuffle=False, num_workers=4)
    MyTestLoader = DataLoader(test_loader, batch_size=1 ,shuffle=False, num_workers=4)
    MyRotTestLoader = DataLoader(rot_test_loader, batch_size=1 ,shuffle=False, num_workers=4)
    
    
    model = Unet_40k(in_ch=4, out_ch=37).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Beginning Training")
    
    validation_losses = []
    train_losses = []
    test_losses = []
    
    best = 1000000
    patience = 0
    patience_limit = 100
    
    for epoch in range(numberOfEpochs):
        running_losses = []
        MAE_losses = []
        
        for batch_idx, batch in enumerate(MyTrainLoader):    
            model.train()
            images = batch['image']
            images = images.permute(2,1,0)
            images = images.to(device)
            labels = batch['label'].to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()
            estimates = model(images)
            estimates= estimates.squeeze()
            loss = 1-dice_coeff(estimates, labels)
            loss.backward()
            optimizer.step()
            running_losses.append(loss.item())
    
       
            
        print('Epoch {} :: Train loss {:.3f}'.format(epoch,np.mean(running_losses)))
    
        train_losses.append(np.mean(running_losses))
    
        if epoch%1 ==0:
            with torch.no_grad():
                running_losses = []
                for batch_idx, batch in enumerate(MyValLoader):    
                    images = batch['image']
                    images = images.permute(2,1,0)
                    images = images.to(device)
                    labels = batch['label'].to(device)
                    labels = labels.squeeze()
                    estimates = model(images)
                    estimates = estimates.squeeze()
                    loss = 1- dice_coeff(estimates, labels)
                    
                    
                    running_losses.append(loss.item())
                    validation_losses.append(np.mean(running_losses))
                val_loss = np.mean(running_losses)
                
                print('Epoch {} :: Valid loss {:.3f}'.format(epoch,np.mean(running_losses)))
                
                
                if val_loss < best:
                    best = val_loss
                    torch.save(model, '/home/lw19/Desktop/final_benchmarking_models/best_Spherical_UNet_Native_warp_rot')
                    patience = 0
                    print('saved_new_best')
                else:
                    patience+=1
                if patience >= patience_limit:
                    if epoch >150:
                        break
                
                print('----------')
    
    torch.save(model, '/home/lw19/Desktop/final_benchmarking_models/final_Spherical_UNet_Native_warp_rot')
    test_loss = []
    model.eval()
    
    for i, batch in enumerate(MyTestLoader):
        test_images = batch['image']
        test_images = test_images.permute(2,1,0)
        test_images = test_images.to(device)
        test_label = batch['label'].to(device)
        estimates = model(test_images)
        loss = dice_coeff(estimates,labels)
        test_loss.append(loss.item())
    
    print('loss', np.mean(test_loss)) 
    
    test_loss = []
    model.eval()
    for i, batch in enumerate(MyRotTestLoader):
        test_images = batch['image']
        test_images = test_images.permute(2,1,0)
        test_images = test_images.to(device)
        test_label = batch['label'].to(device)
        estimates = model(test_images)
        loss = dice_coeff(estimates,labels)
        test_loss.append(loss.item())
    
     
    print('loss', np.mean(test_loss)) 

    return

if __name__ == '__main__':
    main()
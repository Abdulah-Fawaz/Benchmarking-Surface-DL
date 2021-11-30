#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:31:47 2021

@author: logan
"""
import math
import argparse
import sys, os
import numpy as np
import pickle, gzip
import logging
import shutil
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from meshcnn.utils import sparse2tensor, spmatmul, My_Projected_dHCP_Data
from meshcnn.ops import MeshConv
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
mesh_folder='../mesh_files'
numberOfEpochs = 20000

print("Batch Size is ", batch_size)
print("Loss Function is ", criterion)
print("Learning Rate is ", learning_rate)
print("Total Number of Epochs is ", numberOfEpochs)

train_set = np.load('birth_age_confounded/train.npy',allow_pickle=True)
val_set = np.load('birth_age_confounded/validation.npy',allow_pickle=True)
test_set = np.load('birth_age_confounded/test.npy',allow_pickle=True)
    
train_loader = My_Projected_dHCP_Data(train_set, number_of_warps = 99, rotations=True , smoothing = False, 
                                      normalisation='std', parity_choice='both')
val_loader = My_Projected_dHCP_Data(val_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                      normalisation='std', parity_choice='both')
test_loader = My_Projected_dHCP_Data(test_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                     normalisation='std', parity_choice='both')




def make_sampler(arr):
    total = len(arr)
    frac_0 = total/ np.sum(arr[:,-1]>=37)
    weights = np.ones(len(arr)) * frac_0
    frac_1  = total/ np.sum(arr[:,-1]<32)
    frac_2 = total/ (np.sum(arr[:,-1]<37) - np.sum(arr[:,-1]<=32))
    weights[np.where(arr[:,-1]<32)] = frac_1
    weights[np.where(np.logical_and(arr[:,-1]<37 , arr[:,-1]>=32))] = frac_2
    weights = np.tile(weights,2)
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def main():
        
    sampler = make_sampler(train_set)
    
    MyTrainLoader  = DataLoader(train_loader,batch_size=1, shuffle=False, num_workers=2, sampler=sampler)
    MyValLoader  = DataLoader(val_loader,batch_size=1, shuffle=False, num_workers=2)
    MyTestLoader = DataLoader(test_loader, batch_size=1 ,shuffle=False, num_workers=2)
    
    
    rot_test_ds = My_Projected_dHCP_Data(test_set, number_of_warps = 0, rotations=True, smoothing = False, 
                               normalisation='std', parity_choice='both')
    
    MyRotTestLoader =  torch.utils.data.DataLoader(rot_test_ds,1,   shuffle=False, num_workers=1)
    
    model = Model_confound(mesh_folder, feat)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.01)
    
    list_of_all_labels = []
    list_of_all_predictions = []
    overall_results = []
    
    
    list_of_all_rot_labels = []
    list_of_all_rot_predictions = []
    overall_rot_results = []
    
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
            metadata = batch['metadata'].to(device)
            images = images.to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            estimates = model(images,metadata)
            loss = criterion(estimates, labels)
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
                   
                    
                    images = images.to(device)
                    labels = batch['label'].to(device)
                    metadata=batch['metadata'].to(device)
                    
                    estimates = model(images,metadata)
                    loss = criterion(estimates, labels)
                    
                    
                    running_losses.append(loss.item())
                    validation_losses.append(np.mean(running_losses))
                val_loss = np.mean(running_losses)
                
                print('Epoch {} :: Valid loss {:.3f}'.format(epoch,np.mean(running_losses)))
                
                
                if val_loss < best:
                    best = val_loss
                    torch.save(model, '/home/logan/Desktop/final_benchmarking_models/best_UG-SCNN_birth_age_confound_warp_rot')
                    patience = 0
                    print('saved_new_best')
                else:
                    patience+=1
                if patience >= patience_limit:
                    if epoch >150:
                        break
                
                print('----------')
    torch.save(model,'/home/logan/Desktop/final_benchmarking_models/final_UG-SCNN_birth_age_confound_warp_rot')
    test_outputs = []
    test_labels = []
    model.eval()
    for i, batch in enumerate(MyTestLoader):
        test_images = batch['image']
        metadata=batch['metadata'].to(device)
        test_images = test_images.to(device)
        test_label = batch['label'].to(device)
        test_output = model(test_images,metadata)
        test_outputs.append(test_output.item())
        test_labels.append(test_label.item())
    
    print('unrotated average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
    
    overall_results.append(np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))))
    list_of_all_predictions.extend(test_outputs)
    list_of_all_labels.extend(test_labels)
    
    plt.scatter(x = test_labels, y = test_outputs)
    plt.plot(np.arange(30,45), np.arange(30,45))
    plt.show()
    
    test_outputs = []
    test_labels = []
    model.eval()
    
    for i, batch in enumerate(MyRotTestLoader):
        test_images = batch['image']
        metadata=batch['metadata'].to(device)
        test_images = test_images.to(device)
        test_label = batch['label'].to(device)
        test_output = model(test_images,metadata)
        test_outputs.append(test_output.item())
        test_labels.append(test_label.item())
    
    print('rotated average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
    
    overall_rot_results.append(np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))))
    list_of_all_rot_predictions.extend(test_outputs)
    list_of_all_rot_labels.extend(test_labels)
    
    plt.scatter(x = test_labels, y = test_outputs)
    plt.plot(np.arange(30,45), np.arange(30,45))
    plt.show()
    
    # plt.scatter(x = list_of_all_labels, y = list_of_all_predictions)
    # plt.plot(np.arange(25,45), np.arange(25,45))
    # plt.savefig('exp_' + str(experiment) + '_fig_all')
    # plt.show()
    
    # plt.scatter(x = list_of_all_rot_labels, y = list_of_all_rot_predictions)
    # plt.plot(np.arange(25,45), np.arange(25,45))
    # plt.savefig('best_exp_' + str(experiment) + '_fig_all_rot')
    # plt.show()
    
    return

if __name__ == '__main__':
    main()
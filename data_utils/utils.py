#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:43:16 2020

@author: fa19
"""

import numpy as np
import torch
from data_utils.MyDataLoader import My_dHCP_Data, My_dHCP_Data_Graph
from utils import import_from
import matplotlib.pyplot as plt

import sys

def load_dataset_arrays(args):
    
    filename = 'data/'+str(args.dataset_arr)

    train_dataset_arr = np.load( filename + '/train.npy', allow_pickle=True)
    
    validation_dataset_arr = np.load( filename + '/validation.npy', allow_pickle=True)

    test_dataset_arr = np.load( filename + '/test.npy', allow_pickle=True)
    
    return train_dataset_arr, validation_dataset_arr, test_dataset_arr

def load_dataset(train_dataset_arr, validation_dataset_arr, test_dataset_arr ,args):    
#if its a graph add self.edge and then change dataloader
    
    
    train_ds = My_dHCP_Data(train_dataset_arr, projected = args.project, 
                      rotations= args.train_rotations, 
                      parity_choice=args.train_parity, 
                      number_of_warps = args.train_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    
    val_ds = My_dHCP_Data(validation_dataset_arr, projected = args.project, 
                      rotations= args.test_rotations, 
                      parity_choice=args.test_parity, 
                      number_of_warps = args.test_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    
    
    
    test_ds = My_dHCP_Data(test_dataset_arr, projected = args.project, 
                      rotations= False, 
                      parity_choice=args.test_parity, 
                      number_of_warps = args.test_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    

    rot_test_ds = My_dHCP_Data(test_dataset_arr, projected = args.project, 
                      rotations= True, 
                      parity_choice=args.test_parity, 
                      number_of_warps = args.test_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    
    
    return train_ds, val_ds, test_ds, rot_test_ds


def load_dataset_graph(train_dataset_arr, validation_dataset_arr, test_dataset_arr ,args):    
#if its a graph add self.edge and then change dataloader
    edges = torch.LongTensor(np.load('data/edge_ico_6.npy').T)

    
    train_ds = My_dHCP_Data_Graph(train_dataset_arr,edges=edges, projected = args.project, 
                      rotations= args.train_rotations, 
                      parity_choice=args.train_parity, 
                      number_of_warps = args.train_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    
    val_ds = My_dHCP_Data_Graph(validation_dataset_arr, edges = edges, projected = args.project, 
                      rotations= args.test_rotations, 
                      parity_choice=args.test_parity, 
                      number_of_warps = args.test_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    
    
    
    test_ds = My_dHCP_Data_Graph(test_dataset_arr, edges = edges, projected = args.project, 
                      rotations= False, 
                      parity_choice=args.test_parity, 
                      number_of_warps = args.test_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    

    rot_test_ds = My_dHCP_Data_Graph(test_dataset_arr,edges = edges,  projected = args.project, 
                      rotations= True, 
                      parity_choice=args.test_parity, 
                      number_of_warps = args.test_warps,
                      normalisation = args.normalisation,
                      warped_files_directory=args.warp_dir,
                      unwarped_files_directory=args.unwarp_dir)
    
    
    return train_ds, val_ds, test_ds, rot_test_ds

def load_dataloader(ds, dsarr, batch_size = 1, num_workers=1, shuffle = False, weighted = False):
    if weighted == False:
        loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=shuffle, num_workers = num_workers)
        
    elif weighted == True:
        sampler = make_sampler(dsarr)

        loader = torch.utils.data.DataLoader(ds, batch_size, sampler = sampler, num_workers = num_workers)
    
    return loader

def load_dataloader_graph(ds, dsarr, batch_size = 1, num_workers=1, shuffle = False, weighted = False):
    from torch_geometric.data import DataLoader

    if weighted == False:
        loader = DataLoader(ds, batch_size, shuffle=shuffle, num_workers = num_workers)
        
    elif weighted == True:
        sampler = make_sampler(dsarr)
        loader = DataLoader(ds, batch_size, sampler = sampler, num_workers = num_workers)
    
    return loader


def load_dataloader_classification(ds, dsarr, batch_size = 1, num_workers=1, shuffle = False, weighted = False):
    if weighted == False:
        loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=shuffle, num_workers = num_workers)
        
    elif weighted == True:
        sampler = make_classification_sampler(dsarr)

        loader = torch.utils.data.DataLoader(ds, batch_size, sampler = sampler, num_workers = num_workers)
    
    return loader

def load_dataloader_graph_classification(ds, dsarr, batch_size = 1, num_workers=1, shuffle = False, weighted = False):
    from torch_geometric.data import DataLoader

    if weighted == False:
        loader = DataLoader(ds, batch_size, shuffle=shuffle, num_workers = num_workers)
        
    elif weighted == True:
        sampler = make_classification_sampler(dsarr)

        loader = DataLoader(ds, batch_size, sampler = sampler, num_workers = num_workers)
    
    return loader

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
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
    
    return sampler


def make_classification_sampler(arr):
    total = len(arr)
    
    frac_0 = total / np.sum(arr[:,-1] == 0)
    frac_1 = total / np.sum(arr[:,-1] == 1)
    
    
    weights = np.ones(len(arr)) * frac_0
    weights[np.where(arr[:,-1] == 1)] = frac_1
    weights = np.tile(weights,2)
    
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))    
    
    return sampler
    
def load_model(args):
    model_to_load = args.model + '_' + args.task
    model_dir = 'models/'+args.model 
    sys.path.append(model_dir)
    chosen_model = import_from('model', model_to_load)
    return chosen_model



def make_fig(labels, outputs, savedir, name):
    plt.scatter(x = labels, y = outputs)
    plt.savefig(savedir+'/'+str(name))
    plt.close()
    
    
    
    

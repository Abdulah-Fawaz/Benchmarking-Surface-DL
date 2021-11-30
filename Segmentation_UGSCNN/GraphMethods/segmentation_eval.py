#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:01:04 2021

@author: logan
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:57:29 2021

@author: logan
"""
import numpy as np
import math
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import pickle, gzip

import nibabel as nb

import numpy as np
import torch
import random




import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, ChebConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat

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


from graph_dataloader import My_dHCP_Data_Graph_Test, My_dHCP_Data_Graph
#from HCP_graph_dataloader import My_HCP_Data_Graph
from torch_geometric.data import Data



edges = torch.LongTensor(np.load('/data/Data/benchmarking/graph_methods/edge_ico_6.npy').T)

mesh_folder='/data/Data/ugscnn1/mesh_files'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load('/home/lw19/Desktop/final_benchmarking_models/best_GConvNet_TopK_Native_warp_rot')
#model = torch.load('/data/Data/HCP/HCP_models/best_MoUNet_warp_no_rot_upconv_double_conv_DO_2d')

model.to(device)
model.train()

test_set = np.load('/data/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_test_TEA.npy', allow_pickle=True)
#test_set = np.load('/media/logan/Storage/Data/HCP/test_list.npy', allow_pickle=True)


test_ds = My_dHCP_Data_Graph(test_set, 
                      edges = edges, 
                      rotations= False, 
                      parity_choice='left',
                      normalisation = 'std',
                      number_of_warps = 0,
                      output_as_torch=True)

from torch_geometric.data import DataLoader
MyTestLoader = DataLoader(test_ds,batch_size=1, shuffle=False, num_workers=2)

test_loss = []
model.eval()
    #subject_estimates = []
for i, data in enumerate(MyTestLoader):
    data = data.to(device)
    data.y = data.y.to(device)
            
    data.edge_index = data.edge_index.to(device)

    estimates = model(data.x, data.edge_index)
    labels = data.y#.unsqueeze(1)
#    test_labels = test_labels.unsqueeze(1)
    loss = dice_coeff(estimates, labels)
    test_loss.append(loss.item())
print('loss',np.mean(test_loss))
print('std',np.std(test_loss))
print('------')




rot_test_ds = My_dHCP_Data_Graph(test_set, 
                      edges = edges, 
                      rotations= True, 
                      parity_choice='both',
                      normalisation = 'std',
                      number_of_warps = 0,
                      output_as_torch=True)


MyRotTestLoader = DataLoader(rot_test_ds,batch_size=1, shuffle=False, num_workers=2)

test_loss = []

for i, data in enumerate(MyRotTestLoader):
    data = data.to(device)
    data.y = data.y.to(device)
            
    data.edge_index = data.edge_index.to(device)

    estimates = model(data)#, data.edge_index)
    test_labels = data.y#.unsqueeze(1)
    test_labels = test_labels.unsqueeze(1)
    loss = dice_coeff(estimates, test_labels)
  
    test_loss.append(loss.item())

 
print('loss', np.mean(test_loss))
print('std',np.std(test_loss))

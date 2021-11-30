# -*- coding: utf-8 -*-

import numpy as np
import math
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import pickle, gzip
import torch
import nibabel as nb
import numpy as np
import torch
import random
from torch_geometric.data import Data
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, ChebConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat
from graph_dataloader import My_dHCP_Data_Graph
from graph_model import GraphUNet_TopK


train_set = np.load('/data/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_train_TEA.npy', allow_pickle=True)
val_set = np.load('/data/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_val_TEA.npy', allow_pickle=True)
test_set = np.load('/data/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_test_TEA.npy', allow_pickle=True)
edges = torch.LongTensor(np.load('/data/Data/benchmarking/graph_methods/edge_ico_6.npy').T)

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

train_ds = My_dHCP_Data_Graph(train_set,
                      edges=edges, 
                      rotations= False, 
                      parity_choice='both', 
                      number_of_warps = 99, output_as_torch=True,normalisation='std')
    
val_ds = My_dHCP_Data_Graph(val_set, 
                      edges = edges, 
                      rotations= False, 
                      parity_choice='both', 
                      number_of_warps = 0, output_as_torch=True,normalisation='std')

    
    
test_ds = My_dHCP_Data_Graph(test_set, 
                      edges = edges, 
                      rotations= False, 
                      parity_choice='both', 
                      number_of_warps = 0, output_as_torch=True,normalisation='std')

from torch_geometric.data import DataLoader
MyTrainLoader = DataLoader(train_ds,batch_size=1, shuffle=True, num_workers=2)
MyValLoader = DataLoader(val_ds,batch_size=1, shuffle=False, num_workers=2)
MyTestLoader = DataLoader(test_ds,batch_size=1, shuffle=False, num_workers=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GraphUNet_TopK(4,37,4,0.5,False,act=F.relu)
#model = monet_segmentation_deep(num_features=[32,64,128,256,512,1024])
#model = GraphUNet_modded(conv_style=ChebConv,activation_function=nn.ReLU(), in_channels = 4, device=device)

model.to(device)
optimiser = torch.optim.Adam(model.parameters(),lr=1e-4)



validation_losses = []
train_losses = []
test_losses = []

numberOfEpochs = 20000
best = 100000
patience = 0
patience_limit=100

for epoch in range(numberOfEpochs):
    running_losses = []
    MAE_losses = []
    
    for i, data in enumerate(MyTrainLoader):  
        
        data = data.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        model.train()        
        optimiser.zero_grad()
        estimates = model(data.x,data.edge_index)
        labels = data.y#.unsqueeze(1)
        loss = 1-dice_coeff(estimates, labels)
        loss.backward()
        optimiser.step()
        running_losses.append(loss.item())
   
        
    print('Epoch {} :: Train loss {:.3f}'.format(epoch,np.mean(running_losses)))

    train_losses.append(np.mean(running_losses))

    if epoch%1 ==0:
        with torch.no_grad():

          running_losses  = []
          val_outputs = []
          val_labels = []
          for i, data in enumerate(MyValLoader):
              
            data = data.to(device)
            data.y = data.y.to(device)
            data.edge_index = data.edge_index.to(device)
            estimates = model(data.x, data.edge_index)
            labels = data.y
            loss = 1 - dice_coeff(estimates, labels)
            running_losses.append(loss.item())
                
        val_loss = np.mean(running_losses)
        print('validation ', val_loss)
    
            
        if val_loss < best:
            best = val_loss
            torch.save(model,'/home/lw19/Desktop/final_benchmarking_models/best_GConvNet_TopK_Native_warp_no_rot')
            patience = 0
            print('saved_new_best')
        else:
            patience+=1
        if patience >= patience_limit:
            if epoch >150:
                break
        
        print('----------')

torch.save(model,'/home/lw19/Desktop/final_benchmarking_models/final_GConvNet_TopK_Native_warp_no_rot')
test_loss = []
model.eval()
for i, batch in enumerate(MyTestLoader):
    data = data.to(device)
    data.y = data.y.to(device)
            
    data.edge_index = data.edge_index.to(device)

    estimates = model(data.x,data.edge_index)
    labels = data.y#.unsqueeze(1)
#    test_labels = test_labels.unsqueeze(1)
    loss = 1 - dice_coeff(estimates, labels)
    
    test_loss.append(loss.item())

 
print('loss', np.mean(test_loss))

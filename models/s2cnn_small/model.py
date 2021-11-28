#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:58:53 2020

@author: fa19
"""
import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import torch

import numpy as np
from torch.autograd import Variable
import argparse



class s2cnn_dhcp_long(nn.Module):

    def __init__(self, bandwidth=85):
        super(s2cnn_dhcp_long, self).__init__()

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = 3,
                nfeature_out = 16,
                b_in  = bandwidth,
                b_out = bandwidth,
                grid=grid_s2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  =  16,
                nfeature_out = 32,
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 32,
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 64,
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 64,
                nfeature_out = 64,
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 64,
                nfeature_out = 64,
                b_in  = bandwidth//4,
                b_out = bandwidth//8,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 64,
                nfeature_out = 128,
                b_in  = bandwidth//8,
                b_out = bandwidth//8,
                grid=grid_so3_4),
            nn.ReLU(inplace=False)
            )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128,out_features=1),
#            nn.ReLU(inplace=False),
#            # linear 2
#            nn.BatchNorm1d(64),
#            nn.Linear(in_features=64, out_features=32),
#            nn.ReLU(inplace=False),
#            # linear 3
#            nn.BatchNorm1d(32),
#            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x
    
    
    
    
    
    


class s2cnn_dhcp(nn.Module):

    def __init__(self, bandwidth=85):
        super(s2cnn_dhcp, self).__init__()

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = 3,
                nfeature_out = 8,
                b_in  = bandwidth,
                b_out = bandwidth,
                grid=grid_s2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  =  8,
                nfeature_out = 16,
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 16,
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 24,
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 24,
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 32,
                b_in  = bandwidth//4,
                b_out = bandwidth//8,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 64,
                b_in  = bandwidth//8,
                b_out = bandwidth//8,
                grid=grid_so3_4),
            nn.ReLU(inplace=False)
            )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64,out_features=1),
#            nn.ReLU(inplace=False),
#            # linear 2
#            nn.BatchNorm1d(64),
#            nn.Linear(in_features=64, out_features=32),
#            nn.ReLU(inplace=False),
#            # linear 3
#            nn.BatchNorm1d(32),
#            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x
    
    
####################
        
    
class s2cnn_dhcp_long2(nn.Module):

    def __init__(self, bandwidth=85):
        super(s2cnn_dhcp_long, self).__init__()

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = 4,
                nfeature_out = 16,
                b_in  = bandwidth,
                b_out = bandwidth,
                grid=grid_s2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  =  16,
                nfeature_out = 32,
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 32,
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 64,
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 64,
                nfeature_out = 64,
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 64,
                nfeature_out = 64,
                b_in  = bandwidth//4,
                b_out = bandwidth//8,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 64,
                nfeature_out = 128,
                b_in  = bandwidth//8,
                b_out = bandwidth//8,
                grid=grid_so3_4),
            nn.ReLU(inplace=False)
            )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128,out_features=1),
#            nn.ReLU(inplace=False),
#            # linear 2
#            nn.BatchNorm1d(64),
#            nn.Linear(in_features=64, out_features=32),
#            nn.ReLU(inplace=False),
#            # linear 3
#            nn.BatchNorm1d(32),
#            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x
    
    
    
    
    
    


class s2cnn_dhcp2(nn.Module):

    def __init__(self, bandwidth=85):
        super(s2cnn_dhcp2, self).__init__()

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = 4,
                nfeature_out = 8,
                b_in  = bandwidth,
                b_out = bandwidth,
                grid=grid_s2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  =  8,
                nfeature_out = 16,
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 16,
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 24,
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 24,
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 32,
                b_in  = bandwidth//4,
                b_out = bandwidth//8,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 64,
                b_in  = bandwidth//8,
                b_out = bandwidth//8,
                grid=grid_so3_4),
            nn.ReLU(inplace=False)
            )

        self.linear = nn.Sequential(
            # linear 1
            #nn.BatchNorm1d(64),
            nn.Linear(in_features=64,out_features=1),
#            nn.ReLU(inplace=False),
#            # linear 2
#            nn.BatchNorm1d(64),
#            nn.Linear(in_features=64, out_features=32),
#            nn.ReLU(inplace=False),
#            # linear 3
#            nn.BatchNorm1d(32),
#            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x
    
class ResidualBlock(nn.Module):

    def __init__(self, channels1, channels2, bandwidth, beta, shortcut=True):
        super(ResidualBlock, self).__init__()
#        self.inplanes=channels1
        # Exercise 2.1.1 construct the block without shortcut

        res_grid  =  so3_near_identity_grid(n_alpha=6, max_beta=np.pi/beta, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.conv1 = SO3Convolution(
                channels1,
                channels1,
                b_in  = bandwidth,
                b_out = bandwidth,
                grid = res_grid)
        
#        nn.Conv2d(channels1, channels2, kernel_size=3, 
#                               stride=res_stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(channels2)
        self.conv2 = SO3Convolution(
                channels1,
                channels2,
                b_in  = bandwidth,
                b_out = bandwidth // 2,
                grid = res_grid)
        
        
        
        #self.bn2 = nn.BatchNorm2d(channels2)

        if shortcut == True:
        # Exercise 2.1.3 the shortcut; create option for resizing input 
            self.shortcut=nn.Sequential(
                SO3Convolution(
                channels1,
                channels2,
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid = res_grid)
            )
        else:
            self.shortcut=nn.Sequential()
            

    def forward(self, x):
        
        # forward pass: Conv2d > BatchNorm2d > ReLU > 
        #Conv2D >  BatchNorm2d > ADD > ReLU
        out=self.conv1(x)
        #out=self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        # THIS IS WHERE WE ADD THE INPUT
        #print('input shape',x.shape,self.inplanes)
        out += self.shortcut(x)
       # print('res block output shape',  out.shape)
        # final ReLu
        out = F.relu(out)

        return out
   
class s2cnn_small_regression(nn.Module):
    def __init__(self, in_channels, num_features,  num_classes=1, bandwidth = 85, beta = 16):
        '''
        Constructor input:
            block: instance of residual block class
            num_blocks: how many layers per block (used in _make_layer)
            num_strides: list with number of strides for each layer (see Lecture 3)
            num_features: list with number of features for each layer
            FC_Channels: number of inputs expected for the fully connected layer (must
                        equal the total number of activations returned from preceding layer)
            num_classes: (number of outputs of final layer)
        '''
        super(s2cnn_small_regression, self).__init__()

        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=1)
        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)


        
        self.conv2 = S2Convolution(
                nfeature_in  =  num_features[0],
                nfeature_out = num_features[1],
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_s2)

        self.conv3 = SO3Convolution(
                nfeature_in  =  num_features[1],
                nfeature_out = num_features[2],
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_1)
        
        self.conv4 = SO3Convolution(
                nfeature_in  =  num_features[2],
                nfeature_out = num_features[3],
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2)
        
        self.conv5 = SO3Convolution(
                nfeature_in  =  num_features[3],
                nfeature_out = num_features[4],
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3)
        
        

        #self.dropout = nn.Dropout(0.7)
        # ----------------------------------------------------------------------
        #channels1, channels2, current_bandwidth, beta, shortcut=True):
            
            
        self.linear = nn.Sequential(
            # linear 1
            nn.Linear(num_features[4],num_classes))
            
        
    def forward(self, x):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass

        out = F.relu(self.conv1(x))

        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))

        out = so3_integrate(out)
        out = self.linear(out)
        return out
        # ---------------------------------------------------------------------


class s2cnn_small_classification(nn.Module):
    def __init__(self, in_channels, num_features,  num_classes=2, bandwidth = 85, beta = 16):
        '''
        Constructor input:
            block: instance of residual block class
            num_blocks: how many layers per block (used in _make_layer)
            num_strides: list with number of strides for each layer (see Lecture 3)
            num_features: list with number of features for each layer
            FC_Channels: number of inputs expected for the fully connected layer (must
                        equal the total number of activations returned from preceding layer)
            num_classes: (number of outputs of final layer)
        '''
        super(s2cnn_small_classification, self).__init__()

        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=1)
        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        
        
        self.conv2 = S2Convolution(
                nfeature_in  =  num_features[0],
                nfeature_out = num_features[1],
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_s2)

        self.conv3 = SO3Convolution(
                nfeature_in  =  num_features[1],
                nfeature_out = num_features[2],
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_1)
        
        self.conv4 = SO3Convolution(
                nfeature_in  =  num_features[2],
                nfeature_out = num_features[3],
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2)
        
        self.conv5 = SO3Convolution(
                nfeature_in  =  num_features[3],
                nfeature_out = num_features[4],
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3)
        
        self.outac = nn.LogSoftmax(dim=1)

        #self.dropout = nn.Dropout(0.7)
        # ----------------------------------------------------------------------
        #channels1, channels2, current_bandwidth, beta, shortcut=True):
            
            
        self.linear = nn.Sequential(
            # linear 1
            nn.Linear(num_features[4],num_classes))
            
        
    def forward(self, x):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass

        out = F.relu(self.conv1(x))

        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))

        out = so3_integrate(out)
        out = self.linear(out)
        out = self.outac(out)
        return out
        # ---------------------------------------------------------------------



class s2cnn_small_regression_confounded(nn.Module):
    def __init__(self, in_channels, num_features, block = ResidualBlock, num_classes=1, bandwidth = 85, beta = 16):
        '''
        Constructor input:
            block: instance of residual block class
            num_blocks: how many layers per block (used in _make_layer)
            num_strides: list with number of strides for each layer (see Lecture 3)
            num_features: list with number of features for each layer
            FC_Channels: number of inputs expected for the fully connected layer (must
                        equal the total number of activations returned from preceding layer)
            num_classes: (number of outputs of final layer)
        '''
        super(s2cnn_small_regression_confounded, self).__init__()

        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=1)
        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)


        
        self.conv2 = S2Convolution(
                nfeature_in  =  num_features[0],
                nfeature_out = num_features[1],
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_s2)

        self.conv3 = SO3Convolution(
                nfeature_in  =  num_features[1],
                nfeature_out = num_features[2],
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_1)
        
        self.conv4 = SO3Convolution(
                nfeature_in  =  num_features[2],
                nfeature_out = num_features[3],
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2)
        
        self.conv5 = SO3Convolution(
                nfeature_in  =  num_features[3],
                nfeature_out = num_features[4],
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3)

        self.convm = nn.Conv1d(1,4, kernel_size = 1)



        self.linear = nn.Linear(num_features[4] + 4, 500)
        self.linear2 = nn.Linear(500, num_classes)


            
            


    def forward(self, x, m):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass
        out = F.relu(self.conv1(x))

        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))

        out = so3_integrate(out)
        
        



        m = self.convm(m.unsqueeze(2))
        m = nn.ReLU()(m)
        m = m.reshape(m.shape[0],-1)
        
        out = torch.cat([out,m], dim=1)
        
        
        
        out = self.linear(out)
        out = F.relu(out)
        out = self.linear2(out)
        
        
        return out
        # ---------------------------------------------------------------------


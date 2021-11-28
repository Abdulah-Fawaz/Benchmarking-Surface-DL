"""
Created on Wed Sep 16 09:48:23 2020

@author: fa19
"""



import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ResidualBlock import ResidualBlock 
    
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_strides, num_features, in_channels, FC_channels,num_classes=1):
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
        super(ResNet, self).__init__()
        self.in_planes = 32
        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=3, 
                               stride=num_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features[0])
        # num_blocks per layer is given by input argument num_blocks (which is an array)
        self.layer1 = self._make_layer(block, num_features[1], num_blocks, stride=num_strides[1])
        self.layer2 = self._make_layer(block, num_features[2], num_blocks, stride=num_strides[2])
        self.layer3 = self._make_layer(block, num_features[3], num_blocks, stride=num_strides[3])
        self.linear = nn.Linear(FC_channels, num_classes)
        self.dropout = nn.Dropout(0.7)
        # ----------------------------------------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride):

        layers = []
        
        for i in np.arange(num_blocks -1):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes 
        
        layers.append(block(planes, planes, stride))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)

        return out
        # ---------------------------------------------------------------------



class ResNet_2(nn.Module):
    def __init__(self, block, num_blocks, num_strides, num_features, in_channels, FC_channels,num_classes=1):
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
        super(ResNet_2, self).__init__()
        self.in_planes = 32
        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=3, 
                               stride=num_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features[0])
        # num_blocks per layer is given by input argument num_blocks (which is an array)
        self.layer1 = self._make_layer(block, num_features[1], num_blocks, stride=num_strides[1])
        self.layer2 = self._make_layer(block, num_features[2], num_blocks, stride=num_strides[2])
        self.layer3 = self._make_layer(block, num_features[3], num_blocks, stride=num_strides[3])
        self.linear = nn.Linear(FC_channels, num_classes)
        self.dropout = nn.Dropout(0.7)
        # ----------------------------------------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride):

        layers = []
        
        for i in np.arange(num_blocks -1):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes 

        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes 

        return nn.Sequential(*layers)

    def forward(self, x):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
#        out = self.layer2(out)
#        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)

        return out
        # ---------------------------------------------------------------------
        
class presnet_regression(nn.Module):
    def __init__(self, num_features, in_channels, FC_channels = 512*6*6, num_strides = [2,2,2,2,2], block = ResidualBlock, num_blocks = 1, num_classes=1):
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
        super(presnet_regression, self).__init__()
        self.in_planes = 32
        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=3, 
                               stride=num_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features[0])
        # num_blocks per layer is given by input argument num_blocks (which is an array)
        self.layer1 = self._make_layer(block, num_features[1], num_blocks, stride=num_strides[1])
        self.layer2 = self._make_layer(block, num_features[2], num_blocks, stride=num_strides[2])
        self.layer3 = self._make_layer(block, num_features[3], num_blocks, stride=num_strides[3])
        self.layer4 = self._make_layer(block, num_features[4], num_blocks, stride=num_strides[4])

        self.linear = nn.Linear(FC_channels, num_classes)
        self.dropout = nn.Dropout(0.7)
        # ----------------------------------------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride):

        layers = []
        
        for i in np.arange(num_blocks -1):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes 

        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes 

        return nn.Sequential(*layers)

    def forward(self, x):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)

        return out
        # ---------------------------------------------------------------------


class presnet_classification(nn.Module):
    def __init__(self, num_features, in_channels, FC_channels = 512*6*6, num_strides = [2,2,2,2,2], block = ResidualBlock, num_blocks = 1, num_classes=2):
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
        super(presnet_classification, self).__init__()
        self.in_planes = 32
        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=3, 
                               stride=num_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features[0])
        # num_blocks per layer is given by input argument num_blocks (which is an array)
        self.layer1 = self._make_layer(block, num_features[1], num_blocks, stride=num_strides[1])
        self.layer2 = self._make_layer(block, num_features[2], num_blocks, stride=num_strides[2])
        self.layer3 = self._make_layer(block, num_features[3], num_blocks, stride=num_strides[3])
        self.layer4 = self._make_layer(block, num_features[4], num_blocks, stride=num_strides[4])

        self.linear = nn.Linear(FC_channels, num_classes)
        self.dropout = nn.Dropout(0.7)
        self.outac = nn.LogSoftmax(dim=1)
        # ----------------------------------------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride):

        layers = []
        
        for i in np.arange(num_blocks -1):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes 

        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes 

        return nn.Sequential(*layers)

    def forward(self, x):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.outac(out)

        return out
        # ---------------------------------------------------------------------



class presnet_regression_confounded(nn.Module):
    def __init__(self, num_features, in_channels, FC_channels = 512*6*6, num_strides = [2,2,2,2,2], block = ResidualBlock, num_blocks = 1, num_classes=1):
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
        super(presnet_regression_confounded, self).__init__()
        self.in_planes = 32
        # ------------------------------ task 2 -------------------------------
        # complete convolutional and linear layers
        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=3, 
                               stride=num_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features[0])
        # num_blocks per layer is given by input argument num_blocks (which is an array)
        self.layer1 = self._make_layer(block, num_features[1], num_blocks, stride=num_strides[1])
        self.layer2 = self._make_layer(block, num_features[2], num_blocks, stride=num_strides[2])
        self.layer3 = self._make_layer(block, num_features[3], num_blocks, stride=num_strides[3])
        self.layer4 = self._make_layer(block, num_features[4], num_blocks, stride=num_strides[4])

        self.conv11 = nn.Conv1d(1,4, kernel_size = 1)

        self.linear = nn.Linear(FC_channels + 4, num_classes)
        self.dropout = nn.Dropout(0.5)
        # ----------------------------------------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride):

        layers = []
        
        for i in np.arange(num_blocks -1):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes 

        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes 

        return nn.Sequential(*layers)

    def forward(self, x, m):
        # ------------------------------ task 2 -------------------------------
        # complete the forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        
        m = self.conv11(m.unsqueeze(2))
        m = nn.ReLU()(m)
        m = m.reshape(m.shape[0],-1)
        
        out = torch.cat([out,m], dim=1)
        out = self.linear(out)

        return out
        # ---------------------------------------------------------------------

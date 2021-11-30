"""
Created on Wed Sep 16 09:48:23 2020

@author: fa19
"""



import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ResidualBlock import ResidualBlock 

class ResidualBlock(nn.Module):

    def __init__(self, channels1,channels2,res_stride=1):
        super(ResidualBlock, self).__init__()
        self.inplanes=channels1
        # Exercise 2.1.1 construct the block without shortcut
        self.conv1 = nn.Conv2d(channels1, channels2, kernel_size=3, 
                               stride=res_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels2)
        self.conv2 = nn.Conv2d(channels2, channels2, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels2)

        if res_stride != 1 or channels2 != channels1:
        # Exercise 2.1.3 the shortcut; create option for resizing input 
            self.shortcut=nn.Sequential(
                nn.Conv2d(channels1, channels2, kernel_size=1, 
                          stride=res_stride, bias=False),
                nn.BatchNorm2d(channels2)
            )
        else:
            self.shortcut=nn.Sequential()
            

    def forward(self, x):
        
        # forward pass: Conv2d > BatchNorm2d > ReLU > 
        #Conv2D >  BatchNorm2d > ADD > ReLU
        out=self.conv1(x)
        out=self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # THIS IS WHERE WE ADD THE INPUT
        #print('input shape',x.shape,self.inplanes)
        out += self.shortcut(x)
       # print('res block output shape',  out.shape)
        # final ReLu
        out = F.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_strides, num_features, in_channels,num_classes=1):
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
        self.down_layer1 = self._make_layer(block, num_features[1], num_blocks, stride=num_strides[0])
        self.down_layer2 = self._make_layer(block, num_features[2], num_blocks, stride=num_strides[0])
        self.down_layer3 = self._make_layer(block, num_features[3], num_blocks, stride=num_strides[0])
        self.down_layer4 = self._make_layer(block, num_features[4], num_blocks, stride=num_strides[0])
        self.down_layer5 = self._make_layer(block, num_features[5], num_blocks, stride=num_strides[0])
        
        self.up_layer4 = self._make_up_layer(block, num_features[5], num_features[4], num_blocks, stride=num_strides[0])
        self.up_layer3 = self._make_up_layer(block, num_features[4], num_features[3], num_blocks, stride=num_strides[0])
        self.up_layer2 = self._make_up_layer(block, num_features[3], num_features[2], num_blocks, stride=num_strides[0])
        self.up_layer1 = self._make_up_layer(block, num_features[2], num_features[1], num_blocks, stride=num_strides[0])
        self.up_layer0 = self._make_up_layer(block, num_features[1], num_features[0], num_blocks, stride=num_strides[0])
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.dropout = nn.Dropout(0.7)
        self.conv_last = nn.Conv2d(32, 38, 1)
        self.softmax = nn.Softmax(dim=1)
        #-----------------------------------

    def _make_up_layer(self, block, in_chan, out_chan, num_blocks, stride):

        layers = []
        
        for i in np.arange(num_blocks -1):
            layers.append(block(in_chan, out_chan))
            
        layers.append(block(in_chan, out_chan, stride))
        
        return nn.Sequential(*layers)


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
        conv0 = F.relu(self.bn1(self.conv1(x)))
        conv1 = self.down_layer1(conv0)
        conv1 = self.maxpool(conv1)
        
        conv2 = self.down_layer2(conv1)
        conv2 = self.maxpool(conv2)
        
        conv3 = self.down_layer3(conv2)
        conv3 = self.maxpool(conv3)
        
        conv4 = self.down_layer4(conv3)
        conv4 = self.maxpool(conv4)
        
        conv5 = self.down_layer5(conv4)
        conv5 = self.maxpool(conv5)
        
        deconv4 = self.upsample(conv5)
        deconv4 = torch.cat([deconv4, conv4], dim=1)  
        deconv4  = self.up_layer4(deconv4)
        
        deconv3 = self.upsample(deconv4)
        deconv3 = torch.cat([deconv3, conv3], dim=1)  
        deconv3  = self.up_layer3(deconv3)
        
        deconv2 = self.upsample(deconv3)
        deconv2 = torch.cat([deconv2, conv2], dim=1)  
        deconv2  = self.up_layer2(deconv2)
        
        deconv1 = self.upsample(deconv2)
        deconv1 = torch.cat([deconv1, conv1], dim=1)  
        deconv1  = self.up_layer1(deconv1)
        
        deconv0 = self.upsample(deconv1)
        deconv0 = torch.cat([deconv0, conv0], dim=1)  
        deconv0  = self.up_layer0(deconv0)
        
        out = self.conv_last(deconv0)
        out = self.softmax(out)
        


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
class ResNet_3(nn.Module):
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
        super(ResNet_3, self).__init__()
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



def double_conv(in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, bias=False),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.dconv_down1 = ResidualBlock(4, 32)
        self.dconv_down2 = ResidualBlock(32, 64)
        self.dconv_down3 = ResidualBlock(64, 128)
        self.dconv_down4 = ResidualBlock(128, 256)
        self.dconv_down5 = ResidualBlock(256, 512)

        self.avgpool = nn.AvgPool2d(2)
        
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dropout = nn.Dropout2d(0.5)
        self.dconv_up4 = ResidualBlock(256 + 512, 256)
        self.dconv_up3 = ResidualBlock(128 + 256, 128)
        self.dconv_up2 = ResidualBlock(128 + 64, 64)
        self.dconv_up1 = ResidualBlock(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, 37, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        #######   ENCODER ###############
        
        conv1 = self.dconv_down1(x)
        conv1 = self.dropout(conv1)
        x = self.avgpool(conv1)

        # --------------------------------------------------- task 3.3.1 ----------------------------------------------------------
        # implement encoder layers conv2, conv3 and conv4
        
        conv2 = self.dconv_down2(x)
        conv2 = self.dropout(conv2)
        x = self.avgpool(conv2)

        conv3 = self.dconv_down3(x)
        conv3 = self.dropout(conv3)
        x = self.avgpool(conv3)

        conv4 = self.dconv_down4(x)
        conv4 = self.dropout(conv4)
        x = self.avgpool(conv4)

        # --------------------------------------------------- task 3.3.2 ----------------------------------------------------------
        # implement bottleneck
        
        conv5 = self.dconv_down5(x)
        conv5 = self.dropout(conv5)
        # ---------------------------------------------------------------------------------------------------------------------
       
        #######   DECODER ###############
        
        # --------------------------------------------------- task 3.3.3 ----------------------------------------------------------
        # Implement the decoding layers
        
        #deconv4 = self.upsample(conv5)
        self.deconv4_upsample = nn.Upsample(size=(21,21), mode='bilinear')
        deconv4 = self.deconv4_upsample(conv5)
        deconv4 = torch.cat([deconv4, conv4], dim=1)
        deconv4 = self.dconv_up4(deconv4)
        deconv4 = self.dropout(deconv4)

        deconv4 = self.dropout(deconv4)


        deconv3 = self.upsample(deconv4 )       
        deconv3 = torch.cat([deconv3, conv3], dim=1)
        deconv3 = self.dconv_up3(deconv3)
        deconv3 = self.dropout(deconv3)


        self.deconv2_upsample = nn.Upsample(size=(85,85), mode='bilinear')
        deconv2 = self.deconv2_upsample(deconv3)      
        deconv2 = torch.cat([deconv2, conv2], dim=1)
        deconv2 = self.dconv_up2(deconv2)
        deconv2 = self.dropout(deconv2)
       
        deconv1 = self.upsample(deconv2)   
        deconv1 = torch.cat([deconv1, conv1], dim=1)
        deconv1 = self.dconv_up1(deconv1)
        deconv1 = self.dropout(deconv1)

        #---------------------------------------------------------------------------------------------------------------------
        out = self.conv_last(deconv1)
        out = self.softmax(out)

        return out
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:30 2018

@author: zfq

"""

import torch
import numpy as np
import torch.nn as nn
from utils import *
from layers import *
from python_scripts_for_filters_orders.matlab_equivalent_functions import *
import torch.nn.functional as F


class DownNet(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(DownNet, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = my_Get_neighs_order()

        chs = [in_ch, 32, 64, 128, 256, 512]
        
        conv_layer = onering_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])
      
        self.outc = nn.Sequential(
                nn.Linear(chs[5] * 162, out_ch)
                )
                
        
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.outc(x6.view(-1)) # 40962 * 36
        return x
        
class DownNet_mat(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(DownNet_mat, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = my_mat_Get_neighs_order()

        chs = [in_ch, 32, 64, 128, 256, 512, 1024, 2024]
        
        conv_layer = onering_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])
        self.down6 = down_block(conv_layer, chs[5], chs[6], neigh_orders[5], neigh_orders[4])
        self.down7 = down_block(conv_layer, chs[6], chs[7], neigh_orders[6], neigh_orders[5])

        self.outc = nn.Sequential(
                nn.Linear(chs[7] * 12, out_ch)
                )
                
        
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        x = self.outc(x8.flatten()) # 40962 * 36
        return x



class DownNet_mat_2ring_classifier(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, num_classes = 2):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(DownNet_mat_2ring_classifier, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = my_mat_Get_2ring_neighs_order()

        chs = [in_ch,4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        conv_layer = tworing_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])
        self.down6 = down_block(conv_layer, chs[5], chs[6], neigh_orders[5], neigh_orders[4])

        self.outc = nn.Sequential(
                nn.Linear(chs[6] * 42, num_classes)  )
        self.sig = nn.Sigmoid()
                
        
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

#        x = self.outc(x7.view(-1)) # 40962 * 36
        x7 = x7.permute(2,0,1)
        x = self.outc(x7.reshape(x7.shape[0], -1)) 
        x = self.sig(x)
        return x

class DownNet_mat_2ring(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(DownNet_mat_2ring, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = my_mat_Get_2ring_neighs_order()

        chs = [in_ch,16, 32, 64, 128, 256, 512, 1024]
        
        conv_layer = tworing_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])
        self.down6 = down_block(conv_layer, chs[5], chs[6], neigh_orders[5], neigh_orders[4])

        self.outc = nn.Sequential(
                nn.Linear(chs[6] * 42, out_ch)
                )
                
        
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x7 = x7.permute(2,0,1)

        x = self.outc(x7.reshape(x7.shape[0], -1)) 


#        x7 = x7.flatten()
#        x= self.outc(x7)
        return x

class VAE_mat_1ring(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, size):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(VAE_mat_1ring, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = my_mat_Get_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()

        chs = [in_ch, 64 , 64, 128, 256, 512, 2048]
        
        conv_layer = onering_conv_layer
        
        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[1], neigh_orders[0])
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[2], neigh_orders[1])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[5], neigh_orders[4])

        self.fc1 = nn.Linear(chs[5] * 42, size) # logvar
        
        self.fc2 = nn.Linear(chs[5] * 42, size) # mu

        self.fc3 = nn.Linear( size, chs[5] * 42) # joins them back

        self.up1 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
        self.up2 = up_block_no_skip(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
        self.up3 = up_block_no_skip(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
        self.up4 = up_block_no_skip(conv_layer, chs[2], chs[1], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
        self.up5 = up_block_no_skip(conv_layer, chs[1], chs[0], neigh_orders[0], upconv_top_index_40962, upconv_down_index_40962)
        
#        self.outc = nn.Sequential(
#                nn.Linear(chs[1], out_ch)
#                )
                
    def encode(self, x):
        x2 = self.down1(x)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x6 = x6.permute(2,0,1)

        x6 = x6.reshape(x6.shape[0], -1)
        out1 = self.fc1(x6) 
        
        out2 = self.fc2(x6)
        
        

        return out1, out2
    
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        
        eps = torch.randn_like(std)
        
        return mu + eps*std
    
    def decode(self, z):
        z = self.fc3(z)

        deconv_input = z.reshape(-1, 42, int(z.shape[1] / 42))
        deconv_input = deconv_input.permute(1,2,0)
        
        x = self.up1(deconv_input)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x) 
        x = self.up5(x) 
        x = nn.Sigmoid()(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar
    
    def representation(self,x):
       encoding = self.encode(x)
       return self.reparameterize(encoding[0], encoding[1])
   
    
    
    

    
class classify_from_rep(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, size, num_classes):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(classify_from_rep, self).__init__()
       

        self.fc1 = nn.Linear(size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
                
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
 
    
    
class AE_classifier(nn.Module):
    def __init__(self, final_channels, number_of_classes):
        super(AE_classifier, self).__init__()
    
        self.fc = nn.Linear(final_channels * 42, number_of_classes)
    
    def forward(self, encoded_image):
        encoded_image = encoded_image.reshape( -1, 21504)
        out = self.fc(encoded_image)
        
        out = nn.Sigmoid()(out)
        return out
    
    
class AE_mat_1ring(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, size):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(AE_mat_1ring, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = my_mat_Get_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()

        chs = [in_ch, 32 , 64, 128, 256, 512, 2048]
        
        conv_layer = onering_conv_layer
        
        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[1], neigh_orders[0])
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[2], neigh_orders[1])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[5], neigh_orders[4])

#        self.fc1 = nn.Linear(chs[5] * 42, size) # logvar
#        
#        self.fc2 = nn.Linear(chs[5] * 42, size) # mu
#
#        self.fc3 = nn.Linear( size, chs[5] * 42) # joins them back

        self.up1 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
        self.up2 = up_block_no_skip(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
        self.up3 = up_block_no_skip(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
        self.up4 = up_block_no_skip(conv_layer, chs[2], chs[1], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
        self.up5 = up_block_no_skip(conv_layer, chs[1], chs[0], neigh_orders[0], upconv_top_index_40962, upconv_down_index_40962)
        
#        self.outc = nn.Sequential(
#                nn.Linear(chs[1], out_ch)
#                )
                
    def encode(self, x):
        x2 = self.down1(x)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
#        
#        x6 = x6.permute(2,0,1)
#
#        x6 = x6.reshape(x6.shape[0], -1)
#        out1 = self.fc1(x6) 
#        
#        out2 = self.fc2(x6)
        
        
        out = x6
        return out
    

    
    def decode(self, z):
        
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x) 
        x = self.up5(x) 
        x = nn.Sigmoid()(x)
        
        return x
    
    def forward(self, x):
        compressed = self.encode(x)
        
        
        return self.decode(compressed)


"""
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

"""

class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()


#        Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
        )
            
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        
        return x

class up_block_no_skip(nn.Module):
    """Define the upsamping block for a VAE
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block_no_skip, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1):
        
        x1 = self.up(x1)

#        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x1)
    
        return x
    
    
    
class up_block(nn.Module):
    """Define the upsamping block in spherica uent
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x
    
    
class Unet_40k(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(Unet_40k, self).__init__()
        
        neigh_orders = my_mat_Get_2ring_neighs_order()
        upconv = my_mat_Get_2ring_upconv_index()
        chs = [in_ch, 32, 64, 128, 256, 512, 1024]
        conv_layer = tworing_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])
        self.up1 = up_block(conv_layer, chs[5], chs[4], neigh_orders[3], upconv[6], upconv[7])
        self.up2 = up_block(conv_layer, chs[4], chs[3], neigh_orders[2], upconv[4], upconv[5])
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders[1], upconv[2], upconv[3])
        self.up4 = up_block(conv_layer, chs[2], chs[1], neigh_orders[0], upconv[0], upconv[1])
        self.softmax = nn.Softmax(dim=1)
        self.outc = nn.Linear(chs[1], out_ch)

    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)# 40962 * 32
        x = self.outc(x.reshape(40962,32)) # 40962 * 36
        x = self.softmax(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
     
        neigh_orders = Get_neighs_order()
        chs = [3, 32, 64, 128, 256, 512, 1024]
        conv_layer = DiNe_conv_layer

        sequence = []
        sequence.append(conv_layer(chs[0], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
            
        for l in range(1, len(chs)-1):
            sequence.append(pool_layer(neigh_orders[l-1], 'mean'))
            sequence.append(conv_layer(chs[l], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))
            sequence.append(conv_layer(chs[l+1], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*sequence)    
        self.fc =  nn.Sequential(
                nn.Linear(chs[-1], chs[-1]),
                nn.Linear(chs[-1], 2)
                )

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, 0)
        x = self.fc(x)
        return x


#%% dense unet for generator and discriminator

class dense_block(nn.Module):
    def __init__(self, ch, neigh_orders):
        super(dense_block, self).__init__()
        
        self.conv1 = nn.Sequential(
                nn.BatchNorm1d(ch),
                nn.LeakyReLU(0.2, inplace=True),
                DiNe_conv_layer(ch, ch, neigh_orders)
                )

        self.conv2 = nn.Sequential(
                nn.BatchNorm1d(ch*2),
                nn.LeakyReLU(0.2, inplace=True),
                DiNe_conv_layer(ch*2, ch, neigh_orders)
                )
        self.conv3 = nn.Sequential(
                nn.BatchNorm1d(ch*3),
                nn.LeakyReLU(0.2, inplace=True),
                DiNe_conv_layer(ch*3, ch, neigh_orders)
              
                )
        self.conv4 = nn.Sequential(
                nn.BatchNorm1d(ch*4),
                nn.LeakyReLU(0.2, inplace=True),
                DiNe_conv_layer(ch*4, ch, neigh_orders)
                )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x,x1), 1))
        x3 = self.conv3(torch.cat((x,x1,x2), 1))
        x4 = self.conv4(torch.cat((x,x1,x2,x3), 1))
        return x4

class dense_unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dense_unet, self).__init__()
            
        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        #if conv_type == "RePa":
         #   conv_layer = gCNN_conv_layer
        #if conv_type == "DiNe":
        #conv_layer = DiNe_conv_layer

        self.down1 = down_block(DiNe_conv_layer, in_ch, 64, neigh_orders_40962, None, True)
        self.pool1 = pool_layer(neigh_orders_40962, 'mean')
        self.dense1 = dense_block(64, neigh_orders_10242)
        self.pool2 = pool_layer(neigh_orders_10242, 'mean')
        self.dense2 = dense_block(64, neigh_orders_2562)
        self.pool3 = pool_layer(neigh_orders_2562, 'mean')
        self.dense3 = dense_block(64, neigh_orders_642)
        
        self.up1 = upconv_layer(64, 64, upconv_top_index_2562, upconv_down_index_2562)
        self.dense4 = dense_block(64, neigh_orders_2562)
        self.up2 = upconv_layer(64, 64, upconv_top_index_10242, upconv_down_index_10242)
        self.dense5 = dense_block(64, neigh_orders_10242)
        self.up3 = upconv_layer(64, 64, upconv_top_index_40962, upconv_down_index_40962)
            
        self.outc = nn.Sequential(
                DiNe_conv_layer(64, 64, neigh_orders_40962),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2, inplace=True),
                DiNe_conv_layer(64, out_ch, neigh_orders_40962)
                )
                
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.dense1(self.pool1(x1))
        x3 = self.dense2(self.pool2(x2))
        x = self.dense3(self.pool3(x3))
        
        x = nn.functional.leaky_relu(x3 + self.up1(x), negative_slope=0.2)
        x = self.dense4(x)
        x = nn.functional.leaky_relu(x2 + self.up2(x), negative_slope=0.2)
        x = self.dense5(x)
        x = nn.functional.leaky_relu(x1 + self.up3(x), negative_slope=0.2)
        
        x = self.outc(x)
        return x
    
    
class Dense_Discriminator(nn.Module):
    def __init__(self):
        super(Dense_Discriminator, self).__init__()
     
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
       
        self.model = nn.Sequential(       
                        DiNe_conv_layer(1, 48, neigh_orders_40962),
                        dense_block(48, neigh_orders_40962),
                        pool_layer(neigh_orders_40962, 'mean'),
                        dense_block(48, neigh_orders_10242),
                        pool_layer(neigh_orders_10242, 'mean'),
                        dense_block(48, neigh_orders_2562),
                        pool_layer(neigh_orders_2562, 'mean'),
                        dense_block(48, neigh_orders_642),
                        pool_layer(neigh_orders_642, 'mean'),
                        dense_block(48, neigh_orders_162),
                        pool_layer(neigh_orders_162, 'mean'),
                        dense_block(48, neigh_orders_42),
                        pool_layer(neigh_orders_42, 'mean')
                        )

        self.fc = nn.Linear(12*48, 1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(1, x.size()[0] * x.size()[1])
        x = self.out(self.fc(x))
        
        return x
    
    
class D_RealFake(nn.Module):
    def __init__(self):
        super(D_RealFake, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        self.model = nn.Sequential(
                        DiNe_conv_layer(32, 128, neigh_orders_642),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        pool_layer(neigh_orders_642, 'mean'),
                        DiNe_conv_layer(128, 256, neigh_orders_162),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2, inplace=True),
                        pool_layer(neigh_orders_162, 'mean'), 
                        DiNe_conv_layer(256, 512, neigh_orders_42),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2, inplace=True),
                        pool_layer(neigh_orders_42, 'mean')
                )
        self.fc = nn.Linear(512, 1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x,0)
        x = self.out(self.fc(x))
        
        return x
        
    
class D_Subject(nn.Module):
    def __init__(self, num_subjects):
        super(D_Subject, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        self.model = nn.Sequential(
                        DiNe_conv_layer(32, 128, neigh_orders_642),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        pool_layer(neigh_orders_642, 'mean'),
                        DiNe_conv_layer(128, 256, neigh_orders_162),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2, inplace=True),
                        pool_layer(neigh_orders_162, 'mean'), 
                        DiNe_conv_layer(256, 512, neigh_orders_42),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2, inplace=True),
                        pool_layer(neigh_orders_42, 'mean')
                )
        self.fc = nn.Linear(512, num_subjects)
        
    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x,0, True)
        x = self.fc(x)
        
        return x
    
    
    
#%%   
class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
     
        neigh_orders = Get_neighs_order()
        chs = [3, 32, 64, 128, 256, 512, 1024]
        conv_layer = DiNe_conv_layer

        sequence = []
        sequence.append(conv_layer(chs[0], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
            
        for l in range(1, len(chs)-1):
            sequence.append(pool_layer(neigh_orders[l-1], 'mean'))
            sequence.append(conv_layer(chs[l], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.1, inplace=True))
            sequence.append(conv_layer(chs[l+1], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.1, inplace=True))

        self.model = nn.Sequential(*sequence)    
        self.fc =  nn.Sequential(
                nn.Linear(chs[-1], 2)
                )

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, 0, True)
        x = self.fc(x)
        return x



class res_block(nn.Module):
    def __init__(self, c_in, c_out, neigh_orders, first_in_block=False):
        super(res_block, self).__init__()
        
        self.conv1 = DiNe_conv_layer(c_in, c_out, neigh_orders)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = DiNe_conv_layer(c_out, c_out, neigh_orders)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.first = first_in_block
    
    def forward(self, x):
        res = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.first:
            res = torch.cat((res,res),1)
        x = x + res
        x = self.relu(x)
        
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResNet, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        self.conv1 =  DiNe_conv_layer(in_c, 64, neigh_orders_40962)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU(0.2)
        
        self.pool1 = pool_layer(neigh_orders_40962, 'max')
        self.res1_1 = res_block(64, 64, neigh_orders_10242)
        self.res1_2 = res_block(64, 64, neigh_orders_10242)
        self.res1_3 = res_block(64, 64, neigh_orders_10242)
        
        self.pool2 = pool_layer(neigh_orders_10242, 'max')
        self.res2_1 = res_block(64, 128, neigh_orders_2562, True)
        self.res2_2 = res_block(128, 128, neigh_orders_2562)
        self.res2_3 = res_block(128, 128, neigh_orders_2562)
        
        self.pool3 = pool_layer(neigh_orders_2562, 'max')
        self.res3_1 = res_block(128, 256, neigh_orders_642, True)
        self.res3_2 = res_block(256, 256, neigh_orders_642)
        self.res3_3 = res_block(256, 256, neigh_orders_642)
        
        self.pool4 = pool_layer(neigh_orders_642, 'max')
        self.res4_1 = res_block(256, 512, neigh_orders_162, True)
        self.res4_2 = res_block(512, 512, neigh_orders_162)
        self.res4_3 = res_block(512, 512, neigh_orders_162)
                
        self.pool5 = pool_layer(neigh_orders_162, 'max')
        self.res5_1 = res_block(512, 1024, neigh_orders_42, True)
        self.res5_2 = res_block(1024, 1024, neigh_orders_42)
        self.res5_3 = res_block(1024, 1024, neigh_orders_42)
        
        self.fc = nn.Linear(1024, out_c)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pool1(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        
        x = self.pool2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        
        x = self.pool3(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
                
        x = self.pool4(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        
        x = self.pool5(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        
        x = torch.mean(x, 0, True)
        x = self.fc(x)
        x = self.out(x)
        return x

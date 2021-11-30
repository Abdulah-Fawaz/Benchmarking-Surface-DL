#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:54:44 2021

@author: logan
"""
import numpy as np

import torch.nn.functional as F

from torch import nn

import torch
from layers import *
from utils import *
from python_scripts_for_filters_orders.matlab_equivalent_functions import *

hex_6 = torch.LongTensor(np.load('data/hexagons_6.npy'))
hex_5 = torch.LongTensor(np.load('data/hexagons_5.npy'))
hex_4 = torch.LongTensor(np.load('data/hexagons_4.npy'))
hex_3 = torch.LongTensor(np.load('data/hexagons_3.npy'))
hex_2 = torch.LongTensor(np.load('data/hexagons_2.npy'))
hex_1 = torch.LongTensor(np.load('data/hexagons_1.npy'))



reverse_hex_6 = np.load('data/reverse_hex_6.npy')
reverse_hex_5 = np.load('data/reverse_hex_5.npy')
reverse_hex_4 = np.load('data/reverse_hex_4.npy')
reverse_hex_3 = np.load('data/reverse_hex_3.npy')
reverse_hex_2 = np.load('data/reverse_hex_2.npy')
reverse_hex_1 = np.load('data/reverse_hex_1.npy')



edge_index_6 = torch.LongTensor(np.load('data/edge_index_6.npy'))
edge_index_5 = torch.LongTensor(np.load('data/edge_index_5.npy'))
edge_index_4 = torch.LongTensor(np.load('data/edge_index_4.npy'))
edge_index_3 = torch.LongTensor(np.load('data/edge_index_3.npy'))
edge_index_2 = torch.LongTensor(np.load('data/edge_index_2.npy'))
edge_index_1 = torch.LongTensor(np.load('data/edge_index_1.npy'))

polar_coords = torch.Tensor(np.load('data/ico_6_polar.npy'))

cartesian_cords = torch.Tensor(np.load('data/ico_6_cartesian.npy'))



pseudo_6 = torch.Tensor(np.load('data/relative_coords_6.npy'))
pseudo_5 = torch.Tensor(np.load('data/relative_coords_5.npy'))
pseudo_4 = torch.Tensor(np.load('data/relative_coords_4.npy'))
pseudo_3 = torch.Tensor(np.load('data/relative_coords_3.npy'))
pseudo_2 = torch.Tensor(np.load('data/relative_coords_2.npy'))
pseudo_1 = torch.Tensor(np.load('data/relative_coords_1.npy'))


upsample_6 = torch.LongTensor(np.load('data/upsample_to_ico6.npy'))
upsample_5 = torch.LongTensor(np.load('data/upsample_to_ico5.npy'))
upsample_4 = torch.LongTensor(np.load('data/upsample_to_ico4.npy'))
upsample_3 = torch.LongTensor(np.load('data/upsample_to_ico3.npy'))
upsample_2 = torch.LongTensor(np.load('data/upsample_to_ico2.npy'))
upsample_1 = torch.LongTensor(np.load('data/upsample_to_ico1.npy'))

#############################################################################################

upsamples = [upsample_6, upsample_5, upsample_4, upsample_3, upsample_2, upsample_1]
hexes = [hex_6, hex_5, hex_4, hex_3, hex_2, hex_1]

edge_index_6 = torch.LongTensor(np.load('data/edge_index_6.npy'))
edge_index_5 = torch.LongTensor(np.load('data/edge_index_5.npy'))
edge_index_4 = torch.LongTensor(np.load('data/edge_index_4.npy'))
edge_index_3 = torch.LongTensor(np.load('data/edge_index_3.npy'))
edge_index_2 = torch.LongTensor(np.load('data/edge_index_2.npy'))
edge_index_1 = torch.LongTensor(np.load('data/edge_index_1.npy'))




from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, ChebConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat


from graph_dataloader import My_dHCP_Data_Graph




import nibabel as nb

import numpy as np
import torch
import random
from torch_geometric.data import Data

class new_pooling(nn.Module):
    def __init(self,device):
        super(new_pooling,self).__init__()
        
    def forward(self,x):
        L = int((len(x)+6)/4)
        return x[:L]


class hex_upsample(nn.Module):
    def __init__(self,ico_level, device):
        super(hex_upsample, self).__init__()
        self.upsample = upsamples[ico_level].to(device)
        self.hex = hexes[ico_level].to(device)
        
    def forward(self, x,device):
        limit = int(x.shape[0])
        new_x = torch.zeros(self.hex.shape[0],x.shape[1]).to(device)
        new_x[:limit] = x
        new_x[limit:] = torch.mean(x[self.upsample],dim=1)
        
        return new_x

class GraphUNet_no_TopK(nn.Module):
    def __init__(self, conv_style=GCNConv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'):
        super(GraphUNet_modded, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 4
        #self.conv1 = conv_style(self.in_channels, 32, improved=True)
        #self.conv2 = conv_style(32, 64, improved=True)
        #self.conv3 = conv_style(64, 128,improved=True)
        #self.conv4 = conv_style(128, 256, improved=True)
        #self.conv5 = conv_style(256,512,improved=True)
        #self.conv6 = conv_style(512,512,improved=True)
        
        self.conv1 = conv_style(self.in_channels, 32, K=3)
        self.conv2 = conv_style(32, 64, K=3)
        self.conv3 = conv_style(64, 128,K=3)
        self.conv4 = conv_style(128, 256, K=3)
        self.conv5 = conv_style(256,512,K=3)
        self.conv6 = conv_style(512,512,K=3)
        
        self.upsample1 = hex_upsample(5,self.device)
        self.upsample2 = hex_upsample(4,self.device)
        self.upsample3 = hex_upsample(3,self.device)
        self.upsample4 = hex_upsample(2,self.device)
        self.upsample5 = hex_upsample(1,self.device)
        self.upsample6 = hex_upsample(0,self.device)
        
        #self.dropout = nn.Dropout(p=0.2)
        self.pool = new_pooling()
        #self.conv5 = conv_style(256, 512, K=3)
        #self.conv6 = conv_style(512,1024, K=3)
        
        self.activation_function = activation_function
        
        #self.conv6i = conv_style(1024,512, K=6)
        #self.conv5i = conv_style(512,256, K=5)
        

        #self.conv7 = conv_style(512+256, 256,improved=True)
        #self.conv8 = conv_style(256 + 128, 128, improved=True)
        #self.conv9 = conv_style(128 + 64, 64, improved=True)
        #self.conv10 = conv_style(64+32, 32, improved=True)
        #self.conv11 = conv_style(32 +self.in_channels, 37, improved=True)
        
        self.conv7 = conv_style(512+256, 256,K=3)
        self.conv8 = conv_style(256 + 128, 128, K=3)
        self.conv9 = conv_style(128 + 64, 64, K=3)
        self.conv10 = conv_style(64+32, 32, K=3)
        self.conv11 = conv_style(32 +self.in_channels, 37, K=3)
        
        self.softmax = nn.Softmax(dim=1)
        
        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
    def forward(self, data):
        x0 = data.x
        e = data.edge_index
        conv1 = self.conv1(x0,e) #4 -> 32
        x = self.activation_function(conv1)
        x1= self.pool(x)
      
        
        conv2 = self.conv2(x1,edge_index_5.to(self.device)) # 32 -> 64
        x = self.activation_function(conv2)
        x2 = self.pool(x)
    
        
        conv3 = self.conv3(x2,edge_index_4.to(self.device)) # 64 -> 128
        x = self.activation_function(conv3)
        x3 = self.pool(x)
      
        
        conv4 = self.conv4(x3,edge_index_3.to(self.device)) # 128 -> 256 
        x = self.activation_function(conv4)
        x4 = self.pool(x)
     
        
        conv5 = self.conv5(x4,edge_index_2.to(self.device))
        x = self.activation_function(conv5)
        x5 = self.pool(x)
  
        
        conv6 = self.conv6(x5,edge_index_1.to(self.device))
        x = self.activation_function(conv6)
        
        x = self.upsample2(x, self.device)
        
        x = torch.cat([x, x4], dim=1)
        
        conv7 = self.conv7(x,edge_index_2.to(self.device))
        x = self.activation_function(conv7)
     
        
        x = self.upsample3(x, self.device)
        
        x = torch.cat([x,x3], dim=1)
        
        x = self.conv8(x,edge_index_3.to(self.device))
        x= self.activation_function(x)
     
        
        x = self.upsample4(x, self.device)
        
        x = torch.cat([x,x2], dim=1)
        
        x = self.conv9(x,edge_index_4.to(self.device))
        x = self.activation_function(x)
 
        
        x = self.upsample5(x, self.device)
        
        x = torch.cat([x,x1], dim=1)
        
        x = self.conv10(x,edge_index_5.to(self.device))
        x = self.activation_function(x)
 
        x = self.upsample6(x, self.device)
        
        x = torch.cat([x,x0],dim=1)
        x = self.conv11(x,e)
        x = self.softmax(x)
        
        return x

class GraphUNet_TopK(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUNet_TopK, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

    

        self.down_convs = torch.nn.ModuleList([GCNConv(in_channels,32, improved=True), GCNConv(32,64, improved=True), GCNConv(64,128, improved=True), GCNConv(128, 256, improved=True), GCNConv(256,512,improved=True)])
        #self.down_convs = torch.nn.ModuleList([ChebConv(in_channels,32, K=3), ChebConv(32,64, K=3), ChebConv(64,128, K=3), ChebConv(128,256, K=3), ChebConv(256,512, K=3)])
        self.pools = torch.nn.ModuleList([TopKPooling(32,0.5), TopKPooling(64,0.5), TopKPooling(128,0.5), TopKPooling(256,0.5), TopKPooling(512,0.5)])
        



        self.up_convs = torch.nn.ModuleList([GCNConv(512,256,improved=True),GCNConv(256,128,improved=True), GCNConv(128,64,improved=True), GCNConv(64,32, improved=True), GCNConv(32,out_channels, improved=True)])
        #self.up_convs = torch.nn.ModuleList([ChebConv(512,256,K=3),ChebConv(256,128,K=3), ChebConv(128,64,K=3), ChebConv(64,32, K=3), ChebConv(32,out_channels, K=3)])
        self.conv_out = GCNConv(32,37, improved=True)
        #self.conv_out = ChebConv(32,37,K=3)
        self.softmax = nn.Softmax(dim=1)


        self.reset_parameters()


    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()



    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i-1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i  < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]
            

            up = torch.zeros_like(res)
            #up = torch.cat((up,up),dim=0)
            up[perm] = x[:,:len(res[1])]
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x
        x = self.conv_out(x, edge_index, edge_weight)
        x = self.softmax(x)
        return x


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels,
            self.out_channels, self.depth, self.pool_ratios)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:55:34 2021

@author: logan
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import sys; sys.path.append("../../meshcnn")
from meshcnn.ops import MeshConv, DownSamp, ResBlock
import os

class DownSamp(nn.Module):
    def __init__(self, nv_prev):
        super().__init__()
        self.nv_prev = nv_prev



class ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan, level, mesh_folder):
        super().__init__()
        l = level
        mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(l))
        self.conv1 = MeshConv(in_chan, out_chan, mesh_file=mesh_file, stride=1)
        self.conv2 = MeshConv(out_chan, out_chan, mesh_file=mesh_file, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_chan)
        self.bn2 = nn.BatchNorm1d(out_chan)
        self.nv_prev = self.conv2.nv_prev
        
        
        if in_chan != out_chan:
            self.shortcut = nn.Sequential(
                MeshConv(in_chan, out_chan, mesh_file=mesh_file, stride=2),
                nn.BatchNorm1d(out_chan)
            )
        else:
            self.shortcut.nn.Sequential()
            
            
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        
        out = self.relu(out)
        
        return out 
        
        
class Model(nn.Module):
    def __init__(self, mesh_folder, feat=32):
        super().__init__()
        mf = os.path.join(mesh_folder, "icosphere_6.pkl")
        self.in_conv = MeshConv(4, feat, mesh_file=mf, stride=1)
        self.in_bn = nn.BatchNorm1d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
        self.block1 = ResBlock(in_chan=feat, out_chan=2*feat, level=6, mesh_folder=mesh_folder)
        self.block2 = ResBlock(in_chan=2*feat, out_chan=4*feat, level=5,  mesh_folder=mesh_folder)
        self.block3 = ResBlock(in_chan=4*feat, out_chan=8*feat, level=4, mesh_folder=mesh_folder)
        self.block4 = ResBlock(in_chan=8*feat, out_chan=16*feat, level=3,  mesh_folder=mesh_folder)
        self.avg = nn.MaxPool1d(kernel_size=self.block4.nv_prev) # output shape batch x channels x 1
        self.out_layer = nn.Linear(16*feat, 1)

    def forward(self, x):
        x = self.in_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.squeeze(self.avg(x))
        x = F.dropout(x, training=self.training)
        x = self.out_layer(x)

        return x

        
class Model_confound(nn.Module):
    def __init__(self, mesh_folder, feat=32):
        super().__init__()
        mf = os.path.join(mesh_folder, "icosphere_6.pkl")
        self.in_conv = MeshConv(4, feat, mesh_file=mf, stride=1)
        self.in_bn = nn.BatchNorm1d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
        self.block1 = ResBlock(in_chan=feat, out_chan=2*feat, level=6, mesh_folder=mesh_folder)
        self.block2 = ResBlock(in_chan=2*feat, out_chan=4*feat, level=5,  mesh_folder=mesh_folder)
        self.block3 = ResBlock(in_chan=4*feat, out_chan=8*feat, level=4, mesh_folder=mesh_folder)
        self.block4 = ResBlock(in_chan=8*feat, out_chan=16*feat, level=3,  mesh_folder=mesh_folder)
        self.avg = nn.MaxPool1d(kernel_size=self.block4.nv_prev) # output shape batch x channels x 1
        self.out_layer = nn.Linear(16*feat+4, 1)
        self.conv11 = nn.Conv1d(1,4, kernel_size = 1)        

    def forward(self, x,m):
        x = self.in_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.squeeze(self.avg(x))
        x = F.dropout(x, training=self.training)
        m = self.conv11(m)
        m = nn.ReLU()(m)
        m = m.reshape(m.shape[0],-1)
        m = m.squeeze()
       
        out = torch.cat([x,m], dim=0)
        out = out.squeeze()
        x = self.out_layer(out)

        return x
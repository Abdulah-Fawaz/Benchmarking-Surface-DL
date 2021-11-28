#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:52:21 2021

@author: fa19
"""

import numpy as np

import nibabel as nb


s = 2


h = np.load('/home/fa19/Documents/Benchmarking/data/hexagons_' + str(s) + '.npy')

upper = len(h) # int (( len(h) + 6)/4)
limit = int( (len(h) + 6) / 4)


a = {}

for i in range(upper):
    if i >=limit:
        a[i] = []


for i in range(upper): # for every  point
    if i >= limit: #only nonupsampled points
        row = h[i]
        for item in row:
            if item<limit:
                a[i].append(item)
                
data = list(a.values())


S = np.array(data)


        
        



        
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 21:49:10 2021

@author: logan
"""


import nibabel as nb 

import numpy as np 
full_array_L = []
subjid = np.loadtxt('/media/logan/Storage/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_full_TEA',dtype=str)
#subjid = np.loadtxt('/media/logan/Storage/Data/HCP/full_list',dtype=str)
for subject in subjid:
    filename='/media/logan/Storage/Data/benchmarking/fsaverage_32k_30_01_2021/ico6/{}_L.shape.gii'.format(subject)
    #print(filename)
    im = nb.load(filename)
    subj_array_L = []
    for num in range(0,4):
        image = im.darrays[num].data
        subj_array_L.append(image)
    full_array_L.append(subj_array_L)
    
    
full_array_R = []
subjid = np.loadtxt('/media/logan/Storage/Data/dHCP/spherical_unet/spherical_unet/M-CRIB-S_full_TEA',dtype=str)
#subjid = np.loadtxt('/media/logan/Storage/Data/HCP/full_list',dtype=str)
for subject in subjid:
    filename='/media/logan/Storage/Data/benchmarking/fsaverage_32k_30_01_2021/ico6/{}_R.shape.gii'.format(subject)
    #print(filename)
    im = nb.load(filename)
    subj_array_R = []
    for num in range(0,4):
        image = im.darrays[num].data
        subj_array_R.append(image)
    full_array_R.append(subj_array_R)
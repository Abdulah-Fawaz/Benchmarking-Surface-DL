#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:07:44 2021

@author: fa19
"""

import numpy  as np

import matplotlib.pyplot as plt

ds_arr = np.load('/home/fa19/Documents/Benchmarking/data/bayley/full.npy', allow_pickle=True)

import numpy as np
import torch
import random
from scipy.interpolate import griddata
import nibabel as nb

means_birth_age = torch.Tensor([1.18443463, 0.0348339 , 1.02189593, 0.12738451])
stds_birth_age = torch.Tensor([0.39520042, 0.19205919, 0.37749157, 4.16265044])


means_birth_age_confounded = means_birth_age
stds_birth_age_confounded = stds_birth_age




means_scan_age = torch.Tensor([1.16332048, 0.03618059, 1.01341462, 0.09550486])
stds_scan_age = torch.Tensor([0.39418309, 0.18946538, 0.37818974, 4.04483381])



means_bayley = torch.Tensor([0.03561912, 0.1779468,  1.02368241, 1.30365072, 1.42005161,  1.80373678, 1.0485854,  1.44855442,  0.74604417])
stds_bayley = torch.Tensor([0.19094736,  4.11706815,  0.37789417,  4.61303946,  5.08495779,  4.94774891, 4.72248912, 4.22112396, 4.48455344])





means = means_bayley
stds = stds_bayley


rotation_arr = np.load('data/rotations_array.npy').astype(int)
reversing_arr = np.load('data/reversing_arr.npy')


xy_points = np.load('data/equirectangular_ico_6_points.npy')
xy_points[:,0] = (xy_points[:,0] + 0.1)%1
grid = np.load('data/grid_170_square.npy')


grid_x, grid_y = np.meshgrid(np.linspace(0.02, 0.98, 170), np.linspace(0.02, 0.98, 170))
grid[:,0] = grid_x.flatten()
grid[:,1] = grid_y.flatten()

from scipy.interpolate import griddata
from torch_geometric.data import Data


class My_dHCP_Data(torch.utils.data.Dataset):

    def __init__(self, input_arr, warped_files_directory, unwarped_files_directory, rotations = False,
                 number_of_warps = 0, parity_choice = 'left',  sample_only = True, output_as_torch = True ):
        
        """
        
        A Full Dataset for the dHCP Data. Can include warps, rotations and parity flips.
        
        Fileanme style:
            
            in the array: only 'sub-X-ses-Y'
            but for the filenames themselves
                Left = 'sub-X_ses-Y_L'
                Right = 'sub-X_ses-Y_R'
                if warped:
                    'sub-X_ses-Y_L_W1'
        
        INPUT ARGS:
        
            1. input_arr:
                Numpy array size Nx2 
                FIRST index MUST be the filename (excluding directory AND L or R ) of MERGED nibabel files
                LAST index must be the (float) label 
                (OPTIONAL) Middle index if size 3 (optional) is any confounding metadata (Also float, e.g scan age for predicting birth age)
        
                        
            2 . rotations - boolean: to add rotations or not to add rotations              
            
            3. number of warps to include - INT
                NB WARPED AR INCLUDED AS FILENAME CHANGES. WARP NUMBER X IS WRITTEN AS filename_WX
                NUMBER OF WARPS CANNOT EXCEED NUMBER OF WARPES PRESENT IN FILES 
                
            4. Particy Choice (JMPORTANT!) - defines left and right-ness
            
                If: 'left'- will output ONLY LEFT 
                If: 'both' - will randomly choose L or R
                If 'combined' - will output a combined array (left first), will be eventually read as a file with twice the number of input channels. as they will be stacked together
                
            5. smoothing - boolean, will clip extremal values according to the smoothing_array 
            
            6. normalisation - str. Will normalise according to 'range', 'std' or 'None'
                Range is from -1 to 1
                Std is mean = 0, std = 1
                
            7. output_as_torch - boolean:
                outputs values as torch Tensors if you want (usually yes)
                
                
        """
        
        
        
        
        self.input_arr = input_arr
        
        self.image_files = input_arr[:,0]
        self.label = input_arr[:,-1]
        
            
        self.rotations = rotations
                
    
        self.number_of_warps = number_of_warps
        
        self.parity = parity_choice

        self.sample_only = sample_only
        
        self.output_as_torch = output_as_torch
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.directory = warped_files_directory
        else:
            self.directory = unwarped_files_directory
            
    def __len__(self):
        
        L = len(self.input_arr)
        
        
        if self.number_of_warps !=0:
            if self.sample_only == False:
                L = L*self.number_of_warps
                
        if self.parity == 'both':
            L = 2* L
            
        return L
    
    
    def __test_input_params__(self):
        assert self.input_arr.shape[1] >=2, 'check your input array is a nunpy array of files and labels'
        assert type(self.number_of_warps) == int, "number of warps must be an in integer (can be 0)"
        assert self.parity in ['left', 'both', 'combined'], "parity choice must be either left, combined or both"
        if self.number_of_rotations != 0:
            assert self.rotation_arr != None,'Must specify a rotation file containing rotation vertex ids if rotations are non-zero'       
        assert self.rotations == bool, 'rotations must be boolean'
        assert self.normalisation in [None, 'none', 'std', 'range'], 'Normalisation must be either std or range'
        
    
    def __genfilename__(self,idx, right):
        
        """
        gets the appropriate file based on input parameters on PARITY and on WARPS
        
        """
        # grab raw filename
 
        raw_filename = self.image_files[idx]
    
        # add parity to it. IN THE FORM OF A LIST!  If requries both will output a list of length 2
        filename = []
        
        if self.parity != 'combined':
            if right == True:
                filename.append(raw_filename + '_R')
                
            else:
                
               filename.append(raw_filename + '_L')
           
           
#        if self.parity == 'left':
#            filename.append(raw_filename + '_L')
#            
#        elif self.parity == 'both':
#            coin_flip = random.randint(0,1)
#            if coin_flip == 0:
#                filename.append(raw_filename + '_L')
#            elif coin_flip == 1:
#                filename.append(raw_filename + '_R')
#                right = True
           
          
        if self.parity == 'combined':
            filename.append(raw_filename + '_L')
            filename.append(raw_filename+'_R')
            
        # filename is now a list of the correct filenames.
        
        # now add warps if required
        
        if self.number_of_warps != 0: 
            warp_choice = str(random.randint(0,self.number_of_warps))
            if warp_choice !='0':
                
                filename = [s + '_W'+warp_choice for s in filename ]


        return filename
                
        
            
            
    def __getitem__(self, idx):
        
        """
        First load the images and collect them as numpy arrays
        
        then collect the label
        
        then collect the metadata (though might be None)
        """
        
        if self.parity == 'both':
            T = self.__len__()//2

            idx, right  = idx % T, idx // T
            filename = self.__genfilename__(idx, right)
        else:
            right = False
            filename = self.__genfilename__(idx, right)        
        
        
        
        image_gifti = [nb.load(self.directory + '/'+individual_filename+'.shape.gii').darrays for individual_filename in filename]

        image = []
        if self.rotations == True:
            
            rotation_choice = random.randint(0, len(rotation_arr)-1)
            if rotation_choice !=0:
                for file in image_gifti:
                    image.extend(item.data[rotation_arr[rotation_choice]] for item in file) 
            else:
                for file in image_gifti:
                    image.extend(item.data for item in file)
        else:
            for file in image_gifti:
                image.extend(item.data for item in file)
        

        
        if right == True:
            image = [item[reversing_arr] for item in image]
        
        
        
        ### labels
#        if self.number_of_warps != 0:
#            
#            idx = idx%len(self.input_arr)
#        label = self.label[idx]

        
        ###### metadata grabbing if necessary
        
        label = self.label[idx]

            
        # torchify if required:
        
        
     
        if self.output_as_torch:
            image = torch.Tensor( image )

            label = torch.Tensor( [label] )

        
        
        sample = {'image': image,'label': label}

        return sample

#my_ds = My_dHCP_Data(ds_arr, "", '/home/fa19/Documents/dHCP_Data_merged/merged/', parity_choice='both')
my_ds = My_dHCP_Data(ds_arr, "", '/data/rsn/merged/', parity_choice='both')

mean_counts  = np.zeros([len(my_ds), 9])
var_counts = np.zeros([len(my_ds), 9])
birth_ages = []
for i in range(len(my_ds)):
    
    item = my_ds[i]
    im = item['image']
    ms = []
    vs = []
    for image in im:
        image = image[torch.logical_not(torch.isnan(image))]
        
        ms.append(torch.mean(image))
        vs.append(torch.var(image))
    
    mean_counts[i]= ms
    var_counts[i] = vs


global_mean = np.mean(mean_counts, axis=0)
global_diffs = mean_counts - global_mean
global_diffs_squared = np.square(global_diffs)
diffs_plus_vars = global_diffs_squared + var_counts

global_var = np.sum( diffs_plus_vars, axis=0) / len(mean_counts)
global_stds = np.sqrt(global_var)



print(global_mean)
print(global_stds)



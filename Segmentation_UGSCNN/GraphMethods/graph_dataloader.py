
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:46:33 2021

@author: logan
"""
import torch 
import random
import numpy as np 
import nibabel as nb
import torch.nn.functional as F

###Template space###
unwarped_files_directory = '/data/Data/benchmarking/fsaverage_32k_30_01_2021/ico6'
unwarped_labels_directory = '/data/Data/dHCP/M-CRIB-S/template_space/ico6L'
warped_files_directory = '/data/Data/benchmarking/fsaverage_32k_30_01_2021/ico6_warped'
warped_labels_directory = '/data/Data/dHCP/M-CRIB-S/template_space/ico6L_warp'

###Native space###
#unwarped_files_directory='/data/Data/derivatives_native_ico6_seg/features'
#warped_files_directory='/data/Data/derivatives_native_ico6_seg/features_warp'
#unwarped_labels_directory ='/data/Data/derivatives_native_ico6_seg/labels'
#warped_labels_directory ='/data/Data/derivatives_native_ico6_seg/labels_warp'


from torch_geometric.data import Data
means = np.load('../dHCP_mean_seg.npy')
std = np.load('../dHCP_std_seg.npy')


means = torch.from_numpy(means)
stds = torch.from_numpy(std)

# means_scan_age = torch.Tensor([1.16332048, 0.03618059, 1.01341462, 0.09550486])
# stds_scan_age = torch.Tensor([0.39418309, 0.18946538, 0.37818974, 4.04483381])


test_rotation_arr = np.load('data/remaining_rotations_array.npy').astype(int)
rotation_arr = np.load('data/rotations_array.npy').astype(int)
reversing_arr = np.load('data/reversing_arr.npy')


class My_dHCP_Data_Graph(torch.utils.data.Dataset):
    
    def __init__(self, input_arr, edges, rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, 
                 sample_only = True, output_as_torch = True):
        
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
        
        
        
        self.edges = edges 
        self.input_arr = input_arr
        
        self.image_files = input_arr[:,0]
        self.label = input_arr[:,-1]
        self.edges = edges
            
        self.rotations = rotations
    
        self.number_of_warps = number_of_warps
        
        self.parity = parity_choice
            
        self.smoothing = smoothing
        self.normalisation = normalisation
        self.sample_only = sample_only

        self.output_as_torch = output_as_torch
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.directory = warped_files_directory
        else:
            self.directory = unwarped_files_directory

        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.labels_directory = warped_labels_directory
        else:
            self.labels_directory = unwarped_labels_directory


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
           
           
        #if self.parity == 'left':
         #   filename.append(raw_filename + '_L')
            
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
        label_gifti = [nb.load(self.labels_directory + '/'+individual_filename+'.label.gii').darrays for individual_filename in filename]

        image = []
        label = []
        if self.rotations == True:
            
            rotation_choice = random.randint(0, len(rotation_arr)-1)
            if rotation_choice !=0:
                for file in image_gifti:
                    image.extend(item.data[rotation_arr[rotation_choice]] for item in file) 
                
                for file in label_gifti:
                    label.extend(item.data[rotation_arr[rotation_choice]] for item in file)   
                    
            else:
                for file in image_gifti:
                    image.extend(item.data for item in file)
                for file in label_gifti:
                    label.extend(item.data for item in file)
        else:
            for file in image_gifti:
                image.extend(item.data for item in file)
            for file in label_gifti:
                label.extend(item.data for item in file)

        
        if right == True:
            image = [item[reversing_arr] for item in image]
            label = [item[reversing_arr] for item in label]
        
        
        ### labels
#        if self.number_of_warps != 0:
#            
#            idx = idx%len(self.input_arr)
#        label = self.label[idx]

        
        ###### metadata grabbing if necessary
        
        

            
        
        if self.smoothing != False:
            for i in range(len(image)):
                image[i] = np.clip(image[i], lower_bound[i%len(lower_bound)].item(), upper_bound[i%len(upper_bound)].item())
                
            
        # torchify if required:
        
        
        if self.normalisation != None:
            if self.normalisation == 'std':
                for i in range(len(image)):
                    
                    image[i] = ( image[i] - means[i%len(means)].item( )) / stds[i%len(stds)].item()
            
            elif self.normalisation == 'range':
                for i in range(len(image)):
                    
                    image[i] = image[i] - minima[i%len(minima)].item()
                    image[i] = image[i] / (maxima[i%len(maxima)].item()- minima[i%len(minima)].item())
            
        if self.output_as_torch:
            image = torch.Tensor( image )

            label = torch.Tensor( label )
            label = F.one_hot(label.to(torch.int64),37).contiguous()
            label = label.squeeze()
            #label = label.permute(0,2,1)
            
         
                

                
        
        sample = Data(x = image.permute(1,0), edge_index = self.edges, 
                          y =  label)
            
        

        return sample

class My_dHCP_Data_Graph_Test(torch.utils.data.Dataset):
    
    def __init__(self, input_arr, edges, rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, 
                 sample_only = True, output_as_torch = True):
        
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
        
        
        
        self.edges = edges 
        self.input_arr = input_arr
        
        self.image_files = input_arr[:,0]
        self.label = input_arr[:,-1]
        self.edges = edges
            
        self.rotations = rotations
    
        self.number_of_warps = number_of_warps
        
        self.parity = parity_choice
            
        self.smoothing = smoothing
        self.normalisation = normalisation
        self.sample_only = sample_only

        self.output_as_torch = output_as_torch
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.directory = warped_files_directory
        else:
            self.directory = unwarped_files_directory

        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.labels_directory = warped_labels_directory
        else:
            self.labels_directory = unwarped_labels_directory


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
           
           
        #if self.parity == 'left':
         #   filename.append(raw_filename + '_L')
            
        #elif self.parity == 'both':
         #   coin_flip = random.randint(0,1)
          #  if coin_flip == 0:
           #     filename.append(raw_filename + '_L')
           # elif coin_flip == 1:
           #     filename.append(raw_filename + '_R')
           #     right = True
           
          
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
        label_gifti = [nb.load(self.labels_directory + '/'+individual_filename+'.label.gii').darrays for individual_filename in filename]

        image = []
        label = []
        if self.rotations == True:
            
            rotation_choice = random.randint(1, len(test_rotation_arr)-1)
            if rotation_choice !=0:
                for file in image_gifti:
                    image.extend(item.data[test_rotation_arr[rotation_choice]] for item in file) 
                
                for file in label_gifti:
                    label.extend(item.data[test_rotation_arr[rotation_choice]] for item in file)   
                    
            else:
                for file in image_gifti:
                    image.extend(item.data for item in file)
                for file in label_gifti:
                    label.extend(item.data for item in file)
        else:
            for file in image_gifti:
                image.extend(item.data for item in file)
            for file in label_gifti:
                label.extend(item.data for item in file)

        
        if right == True:
            image = [item[reversing_arr] for item in image]
            label = [item[reversing_arr] for item in label]
        
        
        ### labels
#        if self.number_of_warps != 0:
#            
#            idx = idx%len(self.input_arr)
#        label = self.label[idx]

        
        ###### metadata grabbing if necessary
        
        

            
        
        if self.smoothing != False:
            for i in range(len(image)):
                image[i] = np.clip(image[i], lower_bound[i%len(lower_bound)].item(), upper_bound[i%len(upper_bound)].item())
                
            
        # torchify if required:
        
        
        if self.normalisation != None:
            if self.normalisation == 'std':
                for i in range(len(image)):
                    
                    image[i] = ( image[i] - means[i%len(means)].item( )) / stds[i%len(stds)].item()
            
            elif self.normalisation == 'range':
                for i in range(len(image)):
                    
                    image[i] = image[i] - minima[i%len(minima)].item()
                    image[i] = image[i] / (maxima[i%len(maxima)].item()- minima[i%len(minima)].item())
            
        if self.output_as_torch:
            image = torch.Tensor( image )

            label = torch.Tensor( label )
            label = F.one_hot(label.to(torch.int64),37).contiguous()
            label = label.squeeze()
            #label = label.permute(0,2,1)
            
         
                

                
        
        sample = Data(x = image.permute(1,0), edge_index = self.edges, 
                          y =  label)
            
        

        return sample

import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.utils.data import Dataset
import pickle, gzip
import random
import nibabel as nb
import torch.nn.functional as F

### REGRESSION FILES ###


#unwarped_files_directory  =  '/data/Data/dHCP/data/ico_6_01-12-2020/merged'
#warped_files_directory  =  '/data/Data/dHCP/data/ico_6_01-12-2020/warped_non_msm'
unwarped_files_directory='/data/Data/derivatives_native_ico6'
warped_files_directory='/data/Data/derivatives_native_ico6/features_warps'

###SCAN AGE###
means = torch.Tensor([1.1267, 0.0345, 1.0176, 0.0556])
stds = torch.Tensor([0.3522, 0.1906, 0.3844, 4.0476])
#means = np.load('/home/lw19/Desktop/dHCP_mean_seg.npy')
#std = np.load('/home/lw19/Desktop/dHCP_std_seg.npy')
#means = torch.from_numpy(means)
#stds = torch.from_numpy(std)


###BIRTH AGE### 
#means = torch.Tensor([1.18443463, 0.0348339 , 1.02189593, 0.12738451])
#stds = torch.Tensor([0.39520042, 0.19205919, 0.37749157, 4.16265044])

### TEMPLATE SEGMENTATION FILES ###

unwarped_files_directory = '/data/Data/benchmarking/fsaverage_32k_30_01_2021/ico6'
unwarped_labels_directory = '/data/Data/dHCP/M-CRIB-S/template_space/ico6L'
warped_files_directory = '/data/Data/benchmarking/fsaverage_32k_30_01_2021/ico6_warped'
warped_labels_directory = '/data/Data/dHCP/M-CRIB-S/template_space/ico6L_warp'

## NATIVE SEGMENTATION FILES
unwarped_files_directory='/data/Data/derivatives_native_ico6_seg/features'
warped_files_directory='/data/Data/derivatives_native_ico6_seg/features_warp'
unwarped_labels_directory ='/data/Data/derivatives_native_ico6_seg/labels'
warped_labels_directory ='/data/Data/derivatives_native_ico6_seg/labels_warp'


rotation_arr = np.load('../rotations_array.npy').astype(int)
reversing_arr = np.load('../reversing_arr.npy')
test_rotation_arr = np.load('../unseen_rots.npy').astype(int)


def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor([m.row, m.col])
    v = torch.FloatTensor(m.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))

def spmatmul(den, sp):
    """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape newlen x #V
    """
    batch_size, in_chan, nv = list(den.size())
    new_len = sp.size()[0]
    den = den.permute(2, 1, 0).contiguous().view(nv, -1)
    res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
    return res

def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))
    return lat, long

def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
    """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    ele, azi = xyz2latlong(V)
    nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
    dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    long = np.linspace(-np.pi, np.pi, nlong+1)
    sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
    intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
    s2 = np.array([ele, azi]).T
    sig_s2 = intp(s2).astype(dtype)
    return sig_s2
    
class My_Projected_dHCP_Data(torch.utils.data.Dataset):

    def __init__(self, input_arr, rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, sample_only = True, output_as_torch = True ):
        
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
            
        self.smoothing = smoothing
        self.normalisation = normalisation
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
        if self.number_of_warps != 0:
            
            idx = idx%len(self.input_arr)
        label = self.label[idx]

        
        ###### metadata grabbing if necessary
        
#        label_gifti = [np.load(self.label_directory + '/'+individual_filename+'.func.npy') for individual_filename in filename]

#        label = label_gifti
        
        
        if self.input_arr.shape[1] > 2:
            
            self.metadata = self.input_arr[:,1:-1]

            metadata = self.metadata[idx]

        else:
            metadata = None
            
        
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

            label = torch.Tensor( [label] )
            
            if isinstance(metadata,np.ndarray):
                
                metadata = torch.Tensor( [metadata] )
                
            
        if hasattr(metadata,'shape'):
            sample = {'image': image, 'metadata' : metadata, 'label': label}
        
        else:
            sample = {'image': image,'label': label}

        return sample

class My_Projected_dHCP_Data_Test(torch.utils.data.Dataset):

    def __init__(self, input_arr, rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, sample_only = True, output_as_torch = True ):
        
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
            
        self.smoothing = smoothing
        self.normalisation = normalisation
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
            
            rotation_choice = random.randint(0, len(test_rotation_arr)-1)
            if rotation_choice !=0:
                for file in image_gifti:
                    image.extend(item.data[test_rotation_arr[rotation_choice]] for item in file)
            else:
                for file in image_gifti:
                    image.extend(item.data for item in file)
        else:
            for file in image_gifti:
                image.extend(item.data for item in file)
        
        
        
        ### labels
        if self.number_of_warps != 0:
            
            idx = idx%len(self.input_arr)
        label = self.label[idx]

        
        ###### metadata grabbing if necessary
        
#        label_gifti = [np.load(self.label_directory + '/'+individual_filename+'.func.npy') for individual_filename in filename]

#        label = label_gifti
        
        
        if self.input_arr.shape[1] > 2:
            
            self.metadata = self.input_arr[:,1:-1]

            metadata = self.metadata[idx]

        else:
            metadata = None
            
        
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

            label = torch.Tensor( [label] )
            
            if isinstance(metadata,np.ndarray):
                
                metadata = torch.Tensor( [metadata] )
                
            
        if hasattr(metadata,'shape'):
            sample = {'image': image, 'metadata' : metadata, 'label': label}
        
        else:
            sample = {'image': image,'label': label}

        return sample


class My_Projected_dHCP_Data_Segmentation(torch.utils.data.Dataset):

    def __init__(self, input_arr, rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, sample_only = True, output_as_torch = True ):
        
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
            
        self.smoothing = smoothing
        self.normalisation = normalisation
        self.sample_only = sample_only
        
        self.output_as_torch = output_as_torch
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.directory = warped_files_directory
        else:
            self.directory = unwarped_files_directory
            
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.label_directory = warped_labels_directory
        else:
            self.label_directory = unwarped_labels_directory
            
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
        label_gifti = [nb.load(self.label_directory + '/'+individual_filename+'.label.gii').darrays for individual_filename in filename]

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
        
        
        
        
        
        
        if self.input_arr.shape[1] > 2:
            
            self.metadata = input_arr[:,1:-1]
            
        else:
            self.metadata = None
            
        
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
            label = F.one_hot(label.to(torch.int64), 37).contiguous()
            label = label.permute(0,2,1)
            if self.metadata != None:
                
                metadata = torch.Tensor( [self.metadata] )
                
            
        if self.metadata != None:
            sample = {'image': image, 'metadata' : self.metadata, 'label': label}
        
        else:
            sample = {'image': image,'label': label}

        return sample
    
    
class My_Projected_dHCP_Data_Segmentation_Test(torch.utils.data.Dataset):

    def __init__(self, input_arr, rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, sample_only = True, output_as_torch = True ):
        
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
            
        self.smoothing = smoothing
        self.normalisation = normalisation
        self.sample_only = sample_only
        
        self.output_as_torch = output_as_torch
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.directory = warped_files_directory
        else:
            self.directory = unwarped_files_directory
            
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.label_directory = warped_labels_directory
        else:
            self.label_directory = unwarped_labels_directory
            
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
        label_gifti = [nb.load(self.label_directory + '/'+individual_filename+'.label.gii').darrays for individual_filename in filename]

        image = []
        label = []
        if self.rotations == True:
            
            rotation_choice = random.randint(0, len(test_rotation_arr)-1)
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
        
        
        
        
        
        
        if self.input_arr.shape[1] > 2:
            
            self.metadata = input_arr[:,1:-1]
            
        else:
            self.metadata = None
            
        
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
            label = F.one_hot(label.to(torch.int64), 37).contiguous()
            label = label.permute(0,2,1)
            if self.metadata != None:
                
                metadata = torch.Tensor( [self.metadata] )
                
            
        if self.metadata != None:
            sample = {'image': image, 'metadata' : self.metadata, 'label': label}
        
        else:
            sample = {'image': image,'label': label}

        return sample
    

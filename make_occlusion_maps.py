#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 01:11:58 2021

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:26:15 2021

@author: fa19
"""
import os
import numpy as np

size_ = 'small_modalitychebnet_nopool'
size_ = 'large_brth_agechebnet_nopool'

raw_k_numpy ='/home/fa19/Documents/Benchmarking/results/chebnet_nopool/scan_age/oq2934/K=10_occlusion_small_scan_agechebnet_nopool.npy'

raw_occlusion_results = np.load(raw_k_numpy)
raw_k_file_dir = raw_k_numpy.split('.npy')[0]


if len(raw_occlusion_results.shape) == 1:
    raw_occlusion_results = np.reshape(raw_occlusion_results, [raw_occlusion_results.shape[0],1])
    
    
import nibabel as nb
random_image = nb.load('/home/fa19/Documents/dHCP_Data_merged/merged/CC00589XX21_184000_L.shape.gii')

random_image.darrays.append(random_image.darrays[0])
for i in range(raw_occlusion_results.shape[1]):
    random_image.darrays[i].data = raw_occlusion_results[:,i]

nb.save(random_image, raw_k_file_dir + '_raw.shape.gii')

icosahedron_faces =nb.load('/home/fa19/Downloads/icosahedrons/ico-6.surf.gii')
icosahedron_faces = icosahedron_faces.darrays[1].data


        
import pickle

with open('data/vertices_occlusion_list.pkl', 'rb') as handle:
    vertices_list = pickle.load(handle)
    
    
import nibabel as nb
single_image = nb.load('/home/fa19/Documents/dHCP_Data_merged/merged/CC00589XX21_184000_L.shape.gii')

single_image.darrays = [single_image.darrays[0]]

other_image = nb.load('/home/fa19/Documents/dHCP_Data_merged/merged/CC00589XX21_184000_L.shape.gii')

model_dirs = ['/home/fa19/Documents/Benchmarking/results/chebnet_nopool/scan_age/oq2934/best_model',
              '/home/fa19/Documents/Benchmarking/results/chebnet/scan_age/us1588/best_model',
              '/home/fa19/Documents/Benchmarking/results/gconvnet/scan_age/yk9527/end_model',
              '/home/fa19/Documents/Benchmarking/results/gconvnet_nopool/scan_age/iu0730/end_model',
              '/home/fa19/Documents/Benchmarking/results/monet/scan_age/qb9910/best_model',
              '/home/fa19/Documents/Benchmarking/results/chebnet_nopool/birth_age_confounded/np9030/best_model',
              '/home/fa19/Documents/Benchmarking/results/chebnet/birth_age_confounded/uq3227/end_model', 
              '//home/fa19/Documents/Benchmarking/results/gconvnet/birth_age_confounded/fv8408/end_model',
              '/home/fa19/Documents/Benchmarking/results/gconvnet_nopool/birth_age_confounded/ob6746/end_model',
              '/home/fa19/Documents/Benchmarking/results/monet/birth_age_confounded/uu2012/end_model',
              '/home/fa19/Documents/Benchmarking/results/sphericalunet/scan_age/bo4260/end_model',
              '/home/fa19/Documents/results2/presnet/scan_age/hr3318/end_model',
              '/home/fa19/Documents/Benchmarking/results/s2cnn_small/scan_age/ot2313/end_model']

model_dirs = ['/home/fa19/Documents/Benchmarking/results/monet_polar/scan_age/bf5958/end_model']

def numpy_to_occlusion(ARR, vertices_list = vertices_list):
    if len(ARR.shape) == 1:
        ARR = np.reshape(ARR, [ARR.shape[0],1])
    
    sol = np.zeros(ARR.shape)
    
    for m in range(40962):
        sol[m,:] = np.mean((ARR[vertices_list[m]]), 0)
    return sol.astype(float)


for model_dir in model_dirs:
    raw_path = os.path.split(model_dir)[0]
    for filename in os.listdir(raw_path):
        if 'occlusion' in filename and '.npy' in filename and 'large_scan_age' in filename:
            raw_filename = filename.split('.')[0]
            arr = np.load(raw_path + '/' + filename)
            if len(arr.shape)>1:
                if arr.shape[1] == 5:
                    

                    
                    full_occlusion = arr[:,0]
                    
                    occlusion_full = numpy_to_occlusion(full_occlusion, vertices_list)
                    single_image.darrays[0].data = occlusion_full
                    nb.save(single_image,'visualisations/full_'+ str(raw_filename)+'.shape.gii')
                    
                    single_image.darrays[0].data = arr[:,0]
                    nb.save(single_image,'visualisations/RAW_full_'+ str(raw_filename)+'.shape.gii')

                    
                    modality_occlusion = arr[:,1:]
                    
                    occlusion_modality = numpy_to_occlusion(modality_occlusion, vertices_list)
                    
                    for m in range(4):
                        other_image.darrays[m].data = occlusion_modality[:,m]
                    
                    nb.save(other_image, 'visualisations/modality_'+str(raw_filename)+'.shape.gii')
                    for m in range(4):
                        other_image.darrays[m].data = arr[:,m]
                    nb.save(other_image, 'visualisations/RAW_modality_'+str(raw_filename)+'.shape.gii')
                                    

#for i in range(len(solution)):
#    solution[i] = np.mean(solution[i])
#
#
#    solution = np.array(solution)
#
#    random_image.darrays[m].data = solution.astype(float)
#nb.save(random_image, raw_k_file_dir + '_processed.shape.gii')


#nb.save(random_image, 'original_image_' + str(size_) + '.shape.gii')


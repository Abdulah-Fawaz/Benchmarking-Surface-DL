#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:32:34 2021

@author: fa19
"""
from scipy.interpolate import griddata
import os
import params
from params import gen_id
import sys
import numpy as np
from os.path import abspath, dirname
import torch
import torch.nn as nn
#sys.path.append(dirname(abspath(__file__)))
from utils import pick_criterion, import_from, load_testing

from data_utils.utils import load_model, make_fig

from data_utils.MyDataLoader import My_dHCP_Data_Test_Rot



import json
from json.decoder import JSONDecodeError

from data_utils.MyDataLoader import My_dHCP_Data, My_dHCP_Data_Graph

def get_device(args):
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    torch.cuda.set_device(args.device)
    return device


def main():

    args = params.parse()
    device = get_device(args)
    
    model_dir = '/home/fa19/Documents/Benchmarking/results/presnet/scan_age/xd3453/best_model'
   
    resdir = '/'.join(model_dir.split('/')[:-1])
    
    model_name = args.model
    
    dsarr = args.dataset_arr
    
    location_of_model = 'models/' + model_name
    
    print(model_name)
    print(dsarr)
    task=args.task
    print(task)
    print(device)

        

    chosen_model = load_model(args)
    print('this is chosen model', chosen_model)
    features = [int(item) for item in args.features.split(',')]
    model = chosen_model(in_channels = args.in_channels, num_features = features)

    print('yes')
#    model.load_state_dict(torch.load('/home/fa19/Documents/s2cnn_small_results/s2cnn_small/birth_age_confounded/vj9738/state_dict.pt'))
    model = model.to(device)
    model = torch.load(model_dir).to(device)
    
    model.eval()
    
    T = np.load('/home/fa19/Documents/Benchmarking/data/'+dsarr+'/test.npy', allow_pickle = True)
#    print(model.state_dict())
#    
#    torch.save(model.state_dict(), '/home/fa19/Documents/s2cnn_small_results/s2cnn_small/birth_age_confounded/sy6719/state_dict.pt')
    
    
    rot_test_ds = My_dHCP_Data_Test_Rot(T, projected = args.project, 
                          rotations = True, 
                          parity_choice = 'both', 
                          number_of_warps = 0,
                          normalisation = 'std',
                          warped_files_directory='/home/fa19/Documents/dHCP_Data_merged/Warped',
                          unwarped_files_directory='/home/fa19/Documents/dHCP_Data_merged/merged')
        
    
    
    rot_test_loader = torch.utils.data.DataLoader(rot_test_ds, batch_size=1, shuffle=False, num_workers=1)
#    model = torch.load('/home/fa19/Documents/Benchmarking/results/sphericalunet/bayley/ho2860/best_model').cuda()
    

    test_outputs = []
    test_labels = []

    print(model_name)    

    for i, batch in enumerate(rot_test_loader):
        test_images = batch['image']
        if model_name == 'sphericalunet':
            test_images = test_images.permute(2,1,0)


        test_images = test_images.to(device)

        
        
        test_label = batch['label'].to(device)
    
    #    test_labels = test_labels.unsqueeze(1)
        if task == 'regression':
            test_output = model(test_images)
            
        elif task == 'regression_confounded':
            
            metadata = batch['metadata'].to(device)            
            #print(metadata.shape)

            test_output = model(test_images, metadata)
        
#        print('did one')
        test_outputs.append(test_output.item())
        test_labels.append(test_label.item())
    
#    print('outputs' , test_outputs)
#    print('labels', test_labels)
    MAE =  np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))) 
    np.save(resdir+'/unseen_rots_labels_preds.npy', [test_labels, test_outputs])
    print(MAE, resdir)
    make_fig(test_labels, test_outputs, resdir, 'test_rotated')
    with open(resdir+'/Output_2.txt', "w") as text_file:
            text_file.write("Unseen Rotated MAE: %f \n" % MAE)
    
    
    
if __name__ == '__main__':
    main()
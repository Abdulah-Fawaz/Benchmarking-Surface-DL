#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:32:30 2021

@author: fa19
"""

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

from data_utils.MyDataLoader import My_dHCP_Data_Graph_Test_Rot

from torch_geometric.data import DataLoader

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

    model_dir = '/home/fa19/Documents/Benchmarking/results/gconvnet/scan_age/vw9433/end_model'

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
#    model = chosen_model(in_channels = args.in_channels, num_features = features)
    print('yes')
#    model = model.to(device)
    model = torch.load(model_dir).to(device)
    print(model)
    model.eval()
    
    T = np.load('/home/fa19/Documents/Benchmarking/data/'+dsarr+'/test.npy', allow_pickle = True)
    
        
    edges = torch.LongTensor(np.load('data/edge_ico_6.npy').T)

    
    rot_test_ds = My_dHCP_Data_Graph_Test_Rot(T,edges=edges, projected = False, 
                  rotations= True, 
                  parity_choice='both', 
                  number_of_warps = 0,
                  normalisation = 'std',
                  warped_files_directory='/home/fa19/Documents/dHCP_Data_merged/Warped',
                  unwarped_files_directory='/home/fa19/Documents/dHCP_Data_merged/merged')
        
    
    
    rot_test_loader = DataLoader(rot_test_ds, batch_size=1, shuffle=False, num_workers=1)
#    model = torch.load('/home/fa19/Documents/Benchmarking/results/sphericalunet/bayley/ho2860/best_model').cuda()
    

    test_outputs = []
    test_labels = []
    model.eval()

    for i, data in enumerate(rot_test_loader):
        
         
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        
        if args.task == 'regression_confounded':
            data.metadata = data.metadata.to(device)

        test_output = model(data)
        test_label = data.y#.unsqueeze(1)
            
 
        test_outputs.append(test_output.item())
        test_labels.append(test_label.item())
    
        
    
     
    MAE =  np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))) 
    np.save(resdir+'/unseen_rots_labels_preds.npy', [test_labels, test_outputs])
    print(MAE, resdir)
    make_fig(test_labels, test_outputs, resdir, 'test_rotated')
    with open(resdir+'/Output_2.txt', "w") as text_file:
            text_file.write("Unseen Rotated MAE: %f \n" % MAE)
    
    
    
if __name__ == '__main__':
    main()
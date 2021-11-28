#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:15:51 2021

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:10:22 2021

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:55:30 2021

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



import json
from json.decoder import JSONDecodeError

from data_utils.MyDataLoader import My_dHCP_Data, My_dHCP_Data_Graph

def get_device(args):
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    torch.cuda.set_device(args.device)
    return device



import networkx as nx

from torch_geometric.data import DataLoader


def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return list(nbrs)

model_dirs = ['/home/fa19/Documents/Benchmarking/results/monet_polar/scan_age/bf5958/end_model',
              '/home/fa19/Documents/Benchmarking/results/chebnet_nopool/scan_age/oq2934/best_model',
              '/home/fa19/Documents/Benchmarking/results/chebnet/scan_age/us1588/best_model',
              '/home/fa19/Documents/Benchmarking/results/gconvnet/scan_age/yk9527/end_model',
              '/home/fa19/Documents/Benchmarking/results/gconvnet_nopool/scan_age/iu0730/end_model',
              '/home/fa19/Documents/Benchmarking/results/monet/scan_age/qb9910/best_model']


#model_dirs = ['/home/fa19/Documents/Benchmarking/results/chebnet_nopool/birth_age_confounded/np9030/best_model',
#              '/home/fa19/Documents/Benchmarking/results/chebnet/birth_age_confounded/uq3227/end_model', 
#              '//home/fa19/Documents/Benchmarking/results/gconvnet/birth_age_confounded/fv8408/end_model',
#              '/home/fa19/Documents/Benchmarking/results/gconvnet_nopool/birth_age_confounded/ob6746/end_model',
#              '/home/fa19/Documents/Benchmarking/results/monet/birth_age_confounded/uu2012/end_model']

model_list = ['monet_polar','chebnet_nopool','chebnet', 'gconvnet', 'gconvnet_nopool', 'monet', 'monet_polar' ]
def main():

    args = params.parse()
    device = get_device(args)
    for idx in range(0,2):
        model_dir = model_dirs[idx]
        
        args.model = model_list[idx]
        args.task = 'regression'
    
       
        part_res = model_dir.split('/')
        
        resdir = '/'.join(part_res[:-1])
        
        
        model_name = args.model

        dsarr = 'scan_age'
        
        location_of_model = 'models/' + model_name
        
        print(model_name)
        print(dsarr)
        task=args.task
        print(task)
    
        means = torch.Tensor([1.1267, 0.0345, 1.0176, 0.0556])
        stds = torch.Tensor([0.3522, 0.1906, 0.3844, 4.0476]) 
        
        chosen_model = load_model(args)
        
        model = torch.load(model_dir).to(device)
        model.eval()
        
        T = np.load('/home/fa19/Documents/Benchmarking/data/'+dsarr+'/large_scan_age.npy', allow_pickle = True)
        
        T = T.reshape([1,len(T)])
    
        edges = torch.LongTensor(np.load('data/edge_ico_6.npy').T)
    
    
        test_ds = My_dHCP_Data_Graph(T,edges=edges, projected = False, 
                      rotations= False, 
                      parity_choice='both', 
                      number_of_warps = 0,
                      normalisation = 'std',
                      warped_files_directory='/home/fa19/Documents/dHCP_Data_merged/Warped',
                      unwarped_files_directory='/home/fa19/Documents/dHCP_Data_merged/merged')
            
            
        edges_ico_6 = np.load('data/ico_6_edges.npy', allow_pickle = True)
        
        G = nx.Graph()
        
        G.add_edges_from(edges_ico_6)
        
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    #    model = torch.load('/home/fa19/Documents/Benchmarking/results/sphericalunet/bayley/ho2860/best_model').cuda()
        
    
        occlusion_results = np.zeros([40962,5])
        model.eval()
    
        for i, batch in enumerate(loader):
            original_image = batch.x
        
        
            if model_name == 'sphericalunet':
                
                original_image = original_image.permute(2,1,0)
    
        original_image = original_image.to(device)
        
        original_target = batch.y.item()
        
        
        size_of_hole = 10
        
        import copy
        original_target = model(batch.to(device)).item()

        for i in range(40962):
            
            mask = knbrs(G, i, size_of_hole)
            
            new_batch = copy.deepcopy(batch).to(device)
    
            for m in range(4):
                
                new_batch.x[mask,m] = -1 * means[m] / stds[m]
             
    
    #    test_labels = test_labels.unsqueeze(1)
            if task == 'regression':
                output = model(new_batch).item()
                
                
                
            elif task == 'regression_confounded':
                
                
                
#                data.metadata = data.metadata.to(device)
                #print(metadata.shape)
        
                output = model(new_batch).item()
            
        
            occlusion_results[i,0] = output - original_target        
            if i%10000 == 0:
                print('Completed ', i)
            
        
        for m in range(4):
                
                
            for i in range(40962):
            
                mask = knbrs(G, i, size_of_hole)
                
                new_batch = copy.deepcopy(batch).to(device)
        
         
                new_batch.x[mask,m] = -1 * means[m] / stds[m]           
    
            
    
    
    #    test_labels = test_labels.unsqueeze(1)
                if task == 'regression':
                    output = model(new_batch).item()
                    
                    
                    
                elif task == 'regression_confounded':
                    
#                    data.metadata = data.metadata.to(device)
                    #print(metadata.shape)
            
                    output = model(new_batch).item()
                
            
                occlusion_results[i,m+1] = output - original_target        
                if i%10000 == 0:
                    print('Completed ', i)     
        
        
        np.save(resdir+'/K='+str(size_of_hole)+'_occlusion_large_scan_age' + str(model_name)+'.npy', occlusion_results)

        del model
    
    
if __name__ == '__main__':
    main()
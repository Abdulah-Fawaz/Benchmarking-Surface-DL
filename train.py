#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:16:37 2020

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
from utils import validate, train, pick_criterion, load_optimiser, import_from, load_testing

from data_utils.MyDataLoader import My_dHCP_Data 
from data_utils.utils import load_dataloader, load_dataset,load_dataset_arrays, load_model, make_fig, load_dataloader_classification
import json
from json.decoder import JSONDecodeError

import copy

def make_logdir(args):
    logdir = os.path.expanduser(args.logdir)
    logdir = os.path.join(logdir, args.model, args.task)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, 'used_args.json')
    if not os.path.exists(logfile):
        open(logfile, 'w')


    return logfile

def make_resultsdir(args):
    resdir = os.path.expanduser(args.resdir)
    resdir = os.path.join(resdir,args.model,args.dataset_arr, args.run_id)
    os.makedirs(resdir, exist_ok=True)

    return resdir

def experiment_exists(args, logfile):
    current_args = args.__dict__



    with open(logfile, 'r') as x:
        
        try:
            data = json.load(x)

            if current_args in data['dataset']:
                exists = True

            else:
                exists = False

                with open(logfile, 'w') as outfile:
                    previous_stuff = data['dataset']

                    previous_stuff.append(current_args)

                    json.dump({"dataset" : previous_stuff}, outfile, sort_keys = True, indent = 4)
            
            
        except JSONDecodeError:


            with open(logfile, 'w') as outfile: 
                json.dump({"dataset" : [current_args]}, outfile, sort_keys = True, indent = 4)
            exists = False
            
    return exists
            
#        if current_args in old_data['dataset']:
#            exists = True
#        else:
#            exists = False
#            with open(logfile, 'w') as outfile:
#                
#                json.dump(old_data['dataset'].append(current_args),outfile)
#
#        
            
#        except JSONDecodeError:
#            json.dump(json.dumps({"dataset" : [current_args]}, sort_keys = True,ensure_ascii=True, indent = 4), outfile)
#            
    
    
    
    
#    try:
#        with open(logfile, 'r') as f:
#            data = json.load(f)
#    print('loaded data is', data)
#except JSON
#    if current_args in data:
#        print('its in here')
#    else:
#        data = data.append(list(copy.deepcopy(current_args)))
#        print(data)
#        with open(logfile, 'w') as f:
#            json.dump(data, f)
#            print('written')

        





def get_device(args):
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    torch.cuda.set_device(args.device)
    return device



        
def main():
    
    args = params.parse()
    device = get_device(args)

    logfile = make_logdir(args)
    criterion = pick_criterion(args)

    exists = experiment_exists(args, logfile)
    print('exists is ', exists)
    if exists == True and args.overwrite == False:
        print('Params already located in logfile. Experiment has already been run. Overwrite not set to True. Attempt Terminated' )

    #else:

    resdir = make_resultsdir(args) # directory for results
    train_arr, val_arr, test_arr = load_dataset_arrays(args)
    train_ds, val_ds, test_ds, rot_test_ds = load_dataset(train_arr, val_arr, test_arr, args)
    train_weighted = args.weighted_sampling 
    train_shuffle = 1 - train_weighted
    if args.task != 'classification':
        train_loader = load_dataloader(train_ds,train_arr,  batch_size = args.train_bsize, num_workers=2, shuffle = train_shuffle, weighted=train_weighted) 
        val_loader = load_dataloader(val_ds,val_arr, shuffle = True, num_workers=1)
        test_loader =  load_dataloader(test_ds, test_arr, shuffle = False, num_workers=1)
        rot_test_loader = load_dataloader(rot_test_ds, test_arr, shuffle = False, num_workers=1)
        
    else:
        train_loader = load_dataloader_classification(train_ds,train_arr,  batch_size = args.train_bsize, num_workers=2, shuffle = train_shuffle, weighted=train_weighted) 
        val_loader = load_dataloader_classification(val_ds,val_arr, shuffle = True, num_workers=1)
        test_loader =  load_dataloader_classification(test_ds, test_arr, shuffle = False, num_workers=1)
        rot_test_loader = load_dataloader_classification(rot_test_ds, test_arr, shuffle = False, num_workers=1)
        
    chosen_model = load_model(args)
    
    features = [int(item) for item in args.features.split(',')]
    
    model = chosen_model(in_channels = args.in_channels, num_features = features)
    model.to(args.device)
    optimiser_fun = load_optimiser(args)
    
    optimiser = optimiser_fun(model.parameters(), lr = args.learning_rate,weight_decay = args.weight_decay)
    
    test_function = load_testing(args) 
    current_best = 1000000
    current_patience = 0
    converged = False

    epoch_counter = 0
    
    while converged == False:
        train(args, model, optimiser,criterion, train_loader, device, epoch_counter)
        
        current_patience, current_best , converged = validate(args, model, criterion, val_loader, 
                                                              current_best,current_patience, device, resdir)
        epoch_counter +=1
        
    torch.save(model, resdir+'/end_model')
    
    U_mae_finished, U_labels_finished, U_outputs_finished = test_function(args, model, criterion, test_loader,device)
    R_mae_finished, R_labels_finished, R_outputs_finished = test_function(args, model, criterion, rot_test_loader,device)
    
    model = torch.load(resdir+'/best_model')
    
    U_mae_best, U_labels_best, U_outputs_best = test_function(args, model, criterion, test_loader,device)
    R_mae_best, R_labels_best, R_outputs_best = test_function(args, model, criterion, rot_test_loader,device)
    
    if U_mae_finished <= U_mae_best:
        
        make_fig(U_labels_finished, U_outputs_finished, resdir, 'unrotated')
        make_fig(R_labels_finished, R_outputs_finished, resdir, 'rotated')
        
        with open(resdir+'/Output.txt', "w") as text_file:
            text_file.write("Unrotated MAE: %f \n" % U_mae_finished)
            text_file.write("Rotated MAE: %f \n" % R_mae_finished)
            text_file.write("Winning model was finished")
            
        np.save(resdir+'/U_preds_labels.npy', [U_outputs_finished, U_labels_finished])
        np.save(resdir+'/R_preds_labels.npy', [R_outputs_finished, R_labels_finished])
    else:
        
        make_fig(U_labels_best, U_outputs_best, resdir, 'unrotated')
        make_fig(R_labels_best, R_outputs_best, resdir, 'rotated')
        
        with open(resdir+'/Output.txt', "w") as text_file:
            text_file.write("Unrotated MAE: %f \n" % U_mae_best)
            text_file.write("Rotated MAE: %f \n" % R_mae_best)
            text_file.write("Winning model was best")
        
        np.save(resdir+'/U_preds_labels.npy', [U_outputs_best, U_labels_best])
        np.save(resdir+'/R_preds_labels.npy', [R_outputs_best, R_labels_best])
            
    with open(resdir+'/ParamsUsed.json', "w") as text_file:
        json.dump({"parameters" : [args.__dict__]}, text_file, sort_keys = True, indent = 4)
    
    
    
if __name__== '__main__':
    
    main()
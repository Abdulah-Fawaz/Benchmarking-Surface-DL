#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:40:31 2020

@author: fa19
"""

import argparse
import os

import yaml

import random
import string

def gen_id(l= 2, n = 4):
    letters = string.ascii_lowercase 
    numbers = string.digits
    result_str = ''.join(random.choice(letters) for i in range(l))
    numbers_str = ''.join(random.choice(numbers) for i in range(n))

    return result_str + numbers_str

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def parse(args=None):
    parser = argparse.ArgumentParser(description='Train a model.',
                                     fromfile_prefix_chars='@')
    
    
    ################## Task and Model Defining Params ##############
    
    
    parser.add_argument('--model', '-m', choices = ['presnet', 
                                                    'sphericalunet', 
                                                    'monet', 
                                                    's2cnn', 
                                                    's2cnn_small',
                                                    'chebnet', 
                                                    'chebnet_nopool',
                                                    'gconvnet',
                                                    'gconvnet_nopool',
                                                    'monet_polar',
                                    
                                                    'snug'], default='',
                        help='model name (a function beginning with the same name must exist in models.py)')
    
    parser.add_argument('--task', choices=['regression', 'regression_confounded','classification', 'segmentation'], default='regression',
                        help='model name (a function ending with the same name must exist in models.py)')

    parser.add_argument('--criterion', choices=['L1', 'L2', 'NLL'], default='L2',
                        help='criterion is the chosen loss function')

    parser.add_argument('--dataset_arr', choices=['scan_age', 'birth_age', 'birth_age_confounded','bayley', 'segmentation'], default='',
                        help='Choose which dataset arr file we are using. Must exist with same name as dataset_arrs folder')


    parser.add_argument('--in_channels', type=int, default= 4,
                        help='Number of input channels. Default is 4 for the four modalities.')    
    
    parser.add_argument('--features', type=str, default= "32,64,128,256,512",
                        help='Features.')    
    
    parser.add_argument('--device', type = int, default=0,
                        help='Which CUDA device to run on')



    ######## logging and saving params ###################
    
    parser.add_argument('--logdir', '-ld', type=str, default='logging',
                        help='directory to save models, logs and checkpoints.')
    parser.add_argument('--resdir', '-rd', type=str, default='results',
                        help='directory to save models, logs and checkpoints.')    
    parser.add_argument('--run_id', '-id', type=str, default=gen_id(),
                        help='identifier for this run')
    
    parser.add_argument('--overwrite', type=str2bool, default=False,
                        help='Chooses whether to repeat an experiment if already found in log')    
    
    parser.add_argument('--log_frequency', type = int, default=1,
                        help='Frequency in logging training results')      

    ###############   dataset params ########################
    
    # all
    
    parser.add_argument('--project', '-p', type = str2bool, default=False,
                        help='Project or not project the data?')    

    parser.add_argument('--normalisation', choices = [None, 'std', 'range'], default='std',
                        help='How to noramlise data? None, std or range - keep it std for standard deviation')    
    
    parser.add_argument('--warp_dir', type=str, default='/home/fa19/Documents/dHCP_Data_merged/Warped',
                        help='directory for warped files')
    
    parser.add_argument('--unwarp_dir', type=str, default='/home/fa19/Documents/dHCP_Data_merged/merged',
                        help='directory for unwarped files')
    
 
    #training dataloader

    parser.add_argument('--train_rotations', type = str2bool, default=False,
                        help='Include rotations in training data?')
     
    parser.add_argument('--train_parity', choices = ['left', 'both', 'combined'], default='both',
                        help='Parity of training data. Either only left, randomly choose left or right, or both at once')


    parser.add_argument('--train_warps', type = int, default=100,
                        help='How many warps to use in training data. Datatype: int. Must be <=100(?)')

  
    parser.add_argument('--train_bsize', '-bs', type=int, default=8,
                        help='training batch size')
    

    parser.add_argument('--weighted_sampling',  type= str2bool, default=True,
                        help='Use weighted Sampling?')
    

    #  validation/test dataloader
    
    parser.add_argument('--test_rotations', '-test_rot', type = str2bool, default=False,
                        help='Include rotations in validation and test data?')
     
    parser.add_argument('--test_parity', choices = ['left', 'both', 'combined'], default='both',
                        help='Parity of test and val data. Either only left, randomly choose left or right, or both at once')


    parser.add_argument('--test_warps', type = int, default=0,
                        help='How many warps to use in test and val data. Datatype: int. Must be <=100(?)')
    
    
    
    ##################### training params #############
    
    parser.add_argument('--optimiser', '-o', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='optimizer to use. Choices: adam or sgd')
    
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5,
                        help='learning rate')
 
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.01,
                        help='weight decay')
       
    parser.add_argument('--patience',type=float, default=100,
                        help='Defines convergence. How many epochs to wait for best validation to improve. ')   
    
    

    parser.add_argument('--n_epochs', '-ne', type=int, default=20000,
                        help='number of training epochs')
    

    parser.add_argument('--from_scratch', '-fs', action='store_true', default=False,
                        help='train from scratch (remove any previous checkpoints in logdir)')
#
#
#
#    # model params
#    parser.add_argument('--nchannels', default=1, type=int,
#                        help='Number of input channels')
#    parser.add_argument('--nfilters', default=[16, 16, 32, 32, 64, 64],
#                        type=lambda x: [int(_) for _ in x.split(',')],
#                        help='Number of filters per layer')
#    parser.add_argument('--pool_layers', default=[0, 1, 0, 1, 0, 0],
#                        type=lambda x: [int(_) for _ in x.split(',')],
#                        help='Pooling layer indicator')
#    parser.add_argument('--concat_branches', default=[0, 0, 0, 0, 0, 0],
#                        type=lambda x: [int(_) for _ in x.split(',')],
#                        help='Which layers to concatenate, when using a two-branch network')
#    parser.add_argument('--dropout', '-do', action='store_true', default=False,
#                        help='Use dropout, where applicable')
#    parser.add_argument('--batch_norm', '-bn', action='store_true', default=False,
#                        help='Use batch normalization')
#    parser.add_argument('--batch_renorm', '-brn', action='store_true', default=False,
#                        help='Use batch re-normalization (only if batch_norm == True)')
#    parser.add_argument('--nonlin', '-nl', type=str, default='prelu',
#                        help='Nonlinearity to be used')
#    parser.add_argument('--spectral_pool', '-sp', action='store_true', default=False,
#                        help='Use spectral pooling instead of max-pooling. ')
#    parser.add_argument('--pool', '-p', choices=['wap', 'max', 'avg'], default='wap',
#                        help='Type of pooling.')
#    parser.add_argument('--n_filter_params', '-nfp', type=int, default=0,
#                        help='Number of filter params (if 0, use max, else do spectral linear interpolation for localized filters.)')
#    parser.add_argument('--weighted_sph_avg', '-wsa', action='store_true', default=False,
#                        help='Use sin(lat) to weight averages on sphere.')
#    parser.add_argument('--final_pool', '-fp', choices=['gap', 'max', 'magnitudes', 'all'], default='gap',
#                        help='Final pooling layer: GAP, MAX, or frequency magnitudes?')
#    parser.add_argument('--extra_loss', '-el', action='store_true', default=False,
#                        help='Add extra loss on second branch for two-branch architecture. ')
#    parser.add_argument('--triplet_loss', '-tl', action='store_true', default=False,
#                        help='Use within-batch triplet loss for retrieval')
#
#    parser.add_argument('--round_batches', '-rb', action='store_true', default=False,
#                        help='Make sure batches always have the same size; necessary for triplet_loss')
#
#    parser.add_argument('--no_final_fc', '-nofc', action='store_true', default=False,
#                        help='Do not use a final fully connected layer.')
#    parser.add_argument('--transform_method', '-tm', choices=['naive', 'sep'], default='naive',
#                        help='SH transform method: NAIVE or SEParation of variables')
#    parser.add_argument('--real_inputs', '-ri', action='store_true', default=False,
#                        help='Leverage symmetry when inputs are real.')

    # use given args instead of cmd line, if they exist

    if isinstance(args, list):
        # if list, parse as cmd lines arguments
        args_out = parser.parse_args(args)
    elif args is not None:
        # if dict, set values directly
        args_out = parser.parse_args('')
        for k, v in args.items():
            setattr(args_out, k, v)
    else:
        args_out = parser.parse_args()
    
    
    check(args_out, parser)

    return args_out



def check(args, parser):

    if args.model in ['s2cnn', 'presnet']:
        assert args.project == True
        
    
    
    
    
#    
#    
#    
#    if args.test_only:
#        assert not args.from_scratch
#
#    if args.model == 'two_branch':
#        assert len(args.nfilters) == len(args.concat_branches)
#
#    if args.extra_loss:
#        assert args.model == 'two_branch'
#
#    if args.triplet_loss:
#        assert args.round_batches
#
#    if args.spectral_pool:
#        # should not change default pooling, as it will not be used!
#        assert args.pool == 'wap'

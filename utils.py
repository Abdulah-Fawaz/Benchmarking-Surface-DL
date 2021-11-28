#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:08:25 2020

@author: fa19
"""
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def weighted_mse_loss(input, target, k=1):
    return torch.mean((1+(target<37)* k )* (input - target) ** 2)


def validate(args, model, criterion, valLoader, current_best,patience, device, savedir):
    patience_limit = args.patience
    
    model_name = args.model
    task = args.task

    with torch.no_grad():
        running_losses  = []


        for i, batch in enumerate(valLoader):    
            images = batch['image']
            if model_name == 'sphericalunet':
                images = images.permute(2,1,0)
            
            images = images.to(device)
            
            labels = batch['label'].cuda()
            
            if task == 'regression':
                estimates = model(images)
                
            elif task == 'classification':
                estimates = model(images)
                labels = labels.long().squeeze(1)
            elif task == 'regression_confounded':
                metadata = batch['metadata'].to(device)

                estimates = model(images,metadata)
            
                # else might need to add confound of scan age
           
            if args.task == 'classification':
                loss = criterion()(estimates, labels)
            else:
                loss = criterion()(estimates, labels)

            running_losses.append(loss.item())
                
        val_loss = np.mean(running_losses)
        print('validation ', val_loss)
        if val_loss < current_best:
            current_best = np.mean(running_losses)
            torch.save(model, savedir+'/best_model')
            patience = 0
            print('saved new best')
            
        else:
            patience+=1
            
        if patience != patience_limit:
            return patience, current_best, False
    
        elif patience == patience_limit:
            return patience, current_best, True

def train(args, model, optimiser,criterion, trainLoader, device, epoch_counter):
    model_name = args.model    
    task = args.task
    running_losses = []
    for i, batch in enumerate(trainLoader):    
        model.train()
        images = batch['image']
        
        if model_name == 'sphericalunet':
                images = images.permute(2,1,0)
            
        images = images.to(device)
        
        labels = batch['label'].cuda()
        
        if task == 'regression':
            estimates = model(images)
            
        elif task == 'classification':
            estimates = model(images)
            labels = labels.long().squeeze(1)
    
        elif task == 'regression_confounded':
            
            metadata = batch['metadata'].to(device)            
            #print(metadata.shape)

            estimates = model(images,metadata)
            
        if args.task == 'classification':
            loss = criterion(weight = torch.Tensor([1,3]).cuda())(estimates, labels)
#            loss = criterion()(estimates, labels)

        else:
            loss = criterion()(estimates, labels)
            
        optimiser.zero_grad()

        loss.backward()
        optimiser.step()
        running_losses.append(loss.item())
            
            
    
    if epoch_counter % args.log_frequency == 0:
        print(epoch_counter, np.mean(running_losses))
            
    return epoch_counter
            
def train_graph(args, model, optimiser,criterion, trainLoader,device, epoch_counter):
    task = args.task
    running_losses = []
    for i, data in enumerate(trainLoader):  
        
        
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        
        if args.task == 'regression_confounded':
            data.metadata = data.metadata.to(device)
        
        
            
        model.train()        
        optimiser.zero_grad()

        estimates = model(data)

        labels = data.y#.unsqueeze(1)

        if task == 'classification':
            labels = labels.long()
            
        loss = criterion()(estimates, labels) 

        loss.backward()
        optimiser.step()
        running_losses.append(loss.item())
            
            
    
    if epoch_counter % args.log_frequency == 0:
        print(epoch_counter, np.mean(running_losses))
            
    return epoch_counter


def validate_graph(args, model, criterion, valLoader, current_best,patience, device, savedir):
    patience_limit = args.patience
    
    task = args.task

    with torch.no_grad():
        running_losses  = []

        for i, data in enumerate(valLoader):
            
            data.x = data.x.to(device)
            data.y = data.y.to(device)
            
            data.edge_index = data.edge_index.to(device)



            if args.task == 'regression_confounded':
                data.metadata = data.metadata.to(device)
                
            estimates = model(data)
            labels = data.y#.unsqueeze(1)
            
            if task == 'classification':
                labels = labels.long()
                            
            loss = criterion()(estimates, labels)

            running_losses.append(loss.item())
                
        val_loss = np.mean(running_losses)
        print('validation ', val_loss)
        
    
        if val_loss < current_best:
            current_best = np.mean(running_losses)
            torch.save(model, savedir+'/best_model')
            patience = 0
            print('saved new best')
            
        else:
            patience+=1
            
        if patience != patience_limit:
            return patience, current_best, False
    
        elif patience == patience_limit:
            return patience, current_best, True

      

def pick_criterion(args):

    if args.criterion == 'L2':
        criterion = nn.MSELoss
    elif args.criterion == 'L1':
        criterion =  nn.L1Loss
    elif args.criterion == 'NLL':
        criterion = nn.NLLLoss
    #### TO DO ##### - ADD  OTHER CRITERIONS FOR OTHER TASKS
    
    return criterion

def load_optimiser(args):
    if args.optimiser == 'adam':
        optimiser_fun = torch.optim.Adam
    elif args.optimiser == 'sgd':
        optimiser_fun = torch.optim.SGD
    return optimiser_fun
            


def test_regression(args, model, criterion, testLoader,device):
    model.eval()
    task =  args.task
    model_name = args.model
    test_outputs = []
    test_labels = []
    model.eval()
    for i, batch in enumerate(testLoader):
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
        
    
        test_outputs.append(test_output.item())
        test_labels.append(test_label.item())
    
     
    MAE =  np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))) 
    
    return MAE, test_labels, test_outputs


def test_classification(args, model, criterion, testLoader,device):
    model.eval()
    task =  args.task
    model_name = args.model
    test_predictions = []
    test_labels = []
    model.eval()
    for i, batch in enumerate(testLoader):
        test_images = batch['image']
        if model_name == 'sphericalunet':
            test_images = test_images.permute(2,1,0)


        test_images = test_images.to(device)
        
        test_label = batch['label'].to(device)
    
    #    test_labels = test_labels.unsqueeze(1)
        test_output = model(test_images)
            
        prediction = torch.argmax(test_output)
        
    
        test_predictions.append(prediction.item())
        test_labels.append(test_label.item())
    
     
    numpy_preds = np.array(test_predictions)
    numpy_labels = np.array(test_labels)
    
    matching = (numpy_preds == numpy_labels)
    accuracy = sum(matching)/len(matching)
    failure = 1-accuracy
    return failure, test_labels, test_predictions


def test_regression_graph(args, model, criterion, testLoader,device):
    model.eval()
    
    model_name = args.model
    test_outputs = []
    test_labels = []
    model.eval()
    
    for i, data in enumerate(testLoader): 
        
        
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
    
    return MAE, test_labels, test_outputs
   
    
def test_classification_graph(args, model, criterion, testLoader,device):
    model.eval()
    
    model_name = args.model
    test_predictions = []
    test_labels = []
    model.eval()
    
    for i, data in enumerate(testLoader): 
        
        
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        
        if args.task == 'regression_confounded':
            data.metadata = data.metadata.to(device)

        test_output = model(data)
        test_label = data.y#.unsqueeze(1)
            
 
        prediction = torch.argmax(test_output)
        
    
        test_predictions.append(prediction.item())
        test_labels.append(test_label.item())
    
     
    numpy_preds = np.array(test_predictions)
    numpy_labels = np.array(test_labels)
    
    matching = (numpy_preds == numpy_labels)
    accuracy = sum(matching)/len(matching)
    failure = 1-accuracy
    return failure, test_labels, test_predictions

def load_testing(args):
    if args.task in ['regression', 'regression_confounded']:
        testing_to_load = 'test_regression'
    
    else:
        testing_to_load = 'test_'+args.task

    test_function = import_from('utils', testing_to_load)
    return test_function



def load_testing_graph(args):
    if args.task in ['regression', 'regression_confounded']:
        testing_to_load = 'test_regression_graph'
    
    else:
        testing_to_load = 'test_'+args.task+'_graph'

    test_function = import_from('utils', testing_to_load)
    return test_function



            

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:51:41 2020

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import gzip
import pickle
import numpy as np
from torch.autograd import Variable
import argparse

from my_models import s2cnn_dhcp, s2cnn_dhcp_long, s2cnn_dhcp2, s2cnn_dhcp_long2, ResS2CNN, ResidualBlock
from MyDataLoader import My_Projected_dHCP_Data
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = DEVICE

torch.cuda.set_device(0)


batch_size = 8 # batch size can be any number

learning_rate = 1e-5


def weighted_mse_loss(input, target, k=1 ):
    return torch.mean((1+(target<37)* k )* (input - target) ** 2)

criterion = weighted_mse_loss

numberOfEpochs = 20000 

experiment = 4




test_results = []

rotated_test_results = []




train_losses_list = []
validation_losses_list = []
rotated_validation_losses_list = []

#import copy

list_of_all_labels = []

list_of_all_predictions = []

overall_results = []

overall_rot_results = []

list_of_all_rot_labels = []
list_of_all_rot_predictions = []


full_file_arr = np.load('scan_age_corrected_input_arr.npy', allow_pickle = True)

term_limit = 37

terms = full_file_arr[full_file_arr[:,1] >= term_limit]
preterms = full_file_arr[full_file_arr[:,1] < term_limit]



training_ratio = 0.8 

test_ratio = 0.1

#validation_ratio = 0.1

training_terms, validation_terms, test_terms = a,b,c = np.split(terms, [int(len(terms)*training_ratio), int(len(terms)*(training_ratio+ test_ratio))])

training_preterms, validation_preterms, test_preterms = a,b,c = np.split(preterms, [int(len(preterms)*training_ratio), int(len(preterms)*(training_ratio+ test_ratio))])


train_set = np.vstack([training_terms, training_preterms])

validation_set = np.vstack([validation_terms, validation_preterms])

test_set = np.vstack([test_terms, test_preterms])





train_ds = My_Projected_dHCP_Data(train_set, number_of_warps = 20, rotations=True, smoothing = False, 
                                  normalisation='std', parity_choice='both', projected = True)


validation_ds = My_Projected_dHCP_Data(validation_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                  normalisation='std', parity_choice='left', projected = True)

test_ds = My_Projected_dHCP_Data(test_set, number_of_warps = 0, rotations=False, smoothing = False, 
                                 normalisation='std', parity_choice='left', projected = True)


MyTrainLoader  = torch.utils.data.DataLoader(train_ds,batch_size,   shuffle=True, num_workers=2)


MyValLoader = torch.utils.data.DataLoader(validation_ds, 1 ,shuffle=False, num_workers=2)


MyTestLoader = torch.utils.data.DataLoader(test_ds, 1 ,shuffle=False, num_workers=2) 


    

rot_test_ds = My_Projected_dHCP_Data(test_set, number_of_warps = 0, rotations=True, smoothing = False, 
                           normalisation='std', parity_choice='left', projected = True)

MyRotTestLoader =  torch.utils.data.DataLoader(rot_test_ds,1,   shuffle=False, num_workers=1)


#    model = ResNet(ResidualBlock,2,[2,2,2,2], [32,64,128,256], FC_channels=256*11*11,  in_channels=4).to(device)

model  = ResS2CNN(ResidualBlock, [4,8,16,32,64,128], 1 ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)
print("#params", sum(x.numel() for x in model.parameters()))


    


validation_losses = []
train_losses =[]
test_losses = []

best = 1000000
patience = 0
patience_limit = 50

for epoch in range(numberOfEpochs):
    running_losses  = []

    for i, batch in enumerate(MyTrainLoader):    
        model.train()
        images = batch['image']

        #images = images.reshape(images.size(0), -1)
        
        images = images.to(device)
        labels = batch['label'].cuda()

        estimates = model(images)
        
        loss = criterion(estimates, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        running_losses.append(loss.item())
    if epoch % 1 == 0:
        print(epoch, np.mean(running_losses))
        
    train_losses.append(np.mean(running_losses))

    if epoch%1 ==0:
        with torch.no_grad():
            running_losses  = []
            val_outputs = []
            val_labels = []
            for i, batch in enumerate(MyValLoader):    
                images = batch['image']
                #images = images.reshape(images.size(0), -1)
                
                images = images.to(device)
                labels = batch['label'].cuda()
                
                estimates = model(images)
                
                
                val_labels.append(labels.item())
                
                val_outputs.append(estimates.item())
                                
                loss = criterion(estimates, labels)
    
                running_losses.append(loss.item())
            val_loss = np.mean(running_losses)
            print('validation ', val_loss)
            if val_loss < best:
                best = np.mean(running_losses)
                torch.save(model, 'best_model')
                patience = 0
                print('saved_new_best')
            else:
                patience+=1
            if patience >= patience_limit:
                if epoch >150:
                    break
                




test_outputs = []
test_labels = []
model.eval()
for i, batch in enumerate(MyTestLoader):
    test_images = batch['image']
        
    test_images = test_images.to(device)
    test_label = batch['label'].to(device)

#    test_labels = test_labels.unsqueeze(1)

    test_output = model(test_images)

    test_outputs.append(test_output.item())
    test_labels.append(test_label.item())

 
print('average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
overall_results.append(np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))))
list_of_all_predictions.extend(test_outputs)
list_of_all_labels.extend(test_labels)

plt.scatter(x = test_labels, y = test_outputs)
plt.plot(np.arange(30,45), np.arange(30,45))
plt.show()



test_outputs = []
test_labels = []

for i, batch in enumerate(MyRotTestLoader):
    test_images = batch['image']

    test_images = test_images.to(device)
    test_label = batch['label'].to(device)

#    test_labels = test_labels.unsqueeze(1)

    test_output = model(test_images)

    test_outputs.append(test_output.item())
    test_labels.append(test_label.item())


print('average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
overall_rot_results.append(np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))))
list_of_all_rot_predictions.extend(test_outputs)
list_of_all_rot_labels.extend(test_labels)

plt.scatter(x = test_labels, y = test_outputs)
plt.plot(np.arange(30,45), np.arange(30,45))
plt.show()


    
    
plt.scatter(x = list_of_all_labels, y = list_of_all_predictions)
plt.plot(np.arange(25,45), np.arange(25,45))
plt.savefig('exp_' + str(experiment) + '_fig_all')

plt.close()

np.save('experiment_' + str(experiment) + '_predictions_norot.npy', [list_of_all_labels, list_of_all_predictions])



np.save('exp' + str(experiment) + '_results_norot',overall_results)


plt.scatter(x = list_of_all_rot_labels, y = list_of_all_rot_predictions)
plt.plot(np.arange(25,45), np.arange(25,45))
plt.savefig('exp_' + str(experiment) + '_fig_all_rot')

plt.close()

np.save('experiment_' + str(experiment) + '_predictions_rot.npy', [list_of_all_rot_labels, list_of_all_rot_predictions])


np.save('exp' + str(experiment) + '_results_rot',overall_rot_results)








##########################

list_of_all_labels = []
list_of_all_rot_labels = []
list_of_all_predictions = []

list_of_all_rot_predictions = []

experiment = "best" + str(experiment)
model = torch.load('best_model')

test_outputs = []
test_labels = []
model.eval()
for i, batch in enumerate(MyTestLoader):
    test_images = batch['image']
        
    test_images = test_images.to(device)
    test_label = batch['label'].to(device)

#    test_labels = test_labels.unsqueeze(1)

    test_output = model(test_images)

    test_outputs.append(test_output.item())
    test_labels.append(test_label.item())

 
print('average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
overall_results.append(np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))))
list_of_all_predictions.extend(test_outputs)
list_of_all_labels.extend(test_labels)

plt.scatter(x = test_labels, y = test_outputs)
plt.plot(np.arange(30,45), np.arange(30,45))
plt.show()



test_outputs = []
test_labels = []

for i, batch in enumerate(MyRotTestLoader):
    test_images = batch['image']

    test_images = test_images.to(device)
    test_label = batch['label'].to(device)

#    test_labels = test_labels.unsqueeze(1)

    test_output = model(test_images)

    test_outputs.append(test_output.item())
    test_labels.append(test_label.item())


print('average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
overall_rot_results.append(np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))))
list_of_all_rot_predictions.extend(test_outputs)
list_of_all_rot_labels.extend(test_labels)

plt.scatter(x = test_labels, y = test_outputs)
plt.plot(np.arange(30,45), np.arange(30,45))
plt.close()


    
    
plt.scatter(x = list_of_all_labels, y = list_of_all_predictions)
plt.plot(np.arange(25,45), np.arange(25,45))
plt.savefig('exp_' + str(experiment) + '_fig_all')

plt.close()

np.save('experiment_' + str(experiment) + '_predictions_norot.npy', [list_of_all_labels, list_of_all_predictions])



np.save('exp' + str(experiment) + '_results_norot',overall_results)


plt.scatter(x = list_of_all_rot_labels, y = list_of_all_rot_predictions)
plt.plot(np.arange(25,45), np.arange(25,45))
plt.savefig('exp_' + str(experiment) + '_fig_all_rot')

plt.close()

np.save('experiment_' + str(experiment) + '_predictions_rot.npy', [list_of_all_rot_labels, list_of_all_rot_predictions])


np.save('exp' + str(experiment) + '_results_rot',overall_rot_results)



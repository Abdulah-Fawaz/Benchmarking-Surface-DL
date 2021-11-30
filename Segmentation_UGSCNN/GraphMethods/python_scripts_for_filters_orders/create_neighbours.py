#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:10:49 2020

@author: fa19
"""

import nibabel as nb
import numpy as np
from matlab_equivalent_functions import *

ico_6_root = '/home/fa19/Downloads/icosahedrons/flipped_ico_2.surf.gii'

file = nb.load(ico_6_root)


ico_6_faces = file.darrays[1].data

ico_6_coords = file.darrays[0].data
ico_6_coords = ico_6_coords / 100

#create neighbours
pairs = []

for row in ico_6_faces:
    pairs.append([row[0], row[1]])
    pairs.append([row[0], row[2]])
    pairs.append([row[1], row[2]])



"""

WARNING. DIFFERENCE BETWEEN ORIGINAL AND MY VERSION!

OUR POINT LABELS BEGIN WITH ZERO. THEIRS DOES NOT.

THEY USE 0 AS A PLACEHOLDER FOR AN EMPTY POINT

WE USE -1 INSTEAD



"""

adj_matrix_ico_6 = -1 * np.ones([len(ico_6_coords), 6])

for pair in pairs:
    row_index = pair[0]
    
    if pair[1] not in adj_matrix_ico_6[row_index]:
        if -1 in adj_matrix_ico_6[row_index]:
            column_index = np.where(adj_matrix_ico_6[row_index] == -1)[0]
            adj_matrix_ico_6[row_index, column_index[0]] = pair[1]
   
    row_index = pair[1]
    
    if pair[0] not in adj_matrix_ico_6[row_index]:
        if -1 in adj_matrix_ico_6[row_index]:
            column_index = np.where(adj_matrix_ico_6[row_index] == -1)[0]
            adj_matrix_ico_6[row_index, column_index[0]] = pair[0]    
    

adj_matrix_ico_6_for_nitrc = np.copy(adj_matrix_ico_6);
for i in range(12):
    adj_matrix_ico_6_for_nitrc[i,5] = i;
    
adj_matrix_ico_6_for_nitrc = adj_matrix_ico_6_for_nitrc.astype(int)

adj_mat_order = np.zeros([len(adj_matrix_ico_6_for_nitrc), 6])

for i in range(len(adj_matrix_ico_6_for_nitrc)):
    
    neighbour_coords = ico_6_coords[adj_matrix_ico_6_for_nitrc[i,:], :]
    
    center_point = ico_6_coords[i]
    
    neighs_angle = compute_angles(neighbour_coords, center_point);
    neighs_angle = neighs_angle + np.pi/4;
    neighs_angle = np.mod(neighs_angle, 2*np.pi);
    args = np.argsort(neighs_angle);
    adj_mat_order[i,:] = adj_matrix_ico_6_for_nitrc[i,args];




print(adj_mat_order)


#np.save('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices'+ '/adj_mat_order_6.npy', adj_mat_order)


adj_matrix_ico_6 = adj_matrix_ico_6.astype(int)

#%% 40962 adj_mat and its orders
adj_mat_ico_5 = np.zeros([10242,6])
for i in range(10242):
    for j in range(6):
        delete_neigh = adj_matrix_ico_6[i,j]
        if delete_neigh == -1:
            new_neigh = i
        else:
            neigh_of_delete_neigh = adj_matrix_ico_6[delete_neigh]
            for k in range(6):
                if neigh_of_delete_neigh[k] < 10242 and neigh_of_delete_neigh[k] != i:
                    new_neigh = neigh_of_delete_neigh[k]

        adj_mat_ico_5[i,j] = new_neigh



adj_mat_ico_5 = adj_mat_ico_5.astype(int)
adj_mat_order = np.zeros([len(adj_mat_ico_5), 6])

for i in range(len(adj_mat_ico_5)):
    
    neighs = ico_6_coords[adj_mat_ico_5[i,:], :]
    
    center_pt = ico_6_coords[i]
    neighs_angle = compute_angles(neighs, center_pt)


    neighs_angle = neighs_angle + np.pi/4;
    neighs_angle = np.mod(neighs_angle, 2*np.pi)
    args = np.argsort(neighs_angle);
    adj_mat_order[i,:] = adj_mat_ico_5[i,args]
    

#np.save('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices' + '/adj_mat_order_5.npy', adj_mat_order)


#%% others 
adj_mat_intermediate = adj_mat_ico_5

nums = [2562, 642, 162, 42, 12]

levels = [4,3,2,1,0]

for n in range(len(nums)):
   
    num = nums[n]
    
    adj_mat = np.zeros([num,6])
    
    for i in range(num):
        
        for j in range(6):
            
            delete_neigh = int(adj_mat_intermediate[i,j])
            
            if delete_neigh == i:
                new_neigh = i
            else:
                neigh_of_delete_neigh = adj_mat_intermediate[delete_neigh]
                
                for k in range(6):
                    
                    if neigh_of_delete_neigh[k] < num+1 and neigh_of_delete_neigh[k] != i:
                        new_neigh = neigh_of_delete_neigh[k]

            adj_mat[i,j] = new_neigh
    adj_mat = adj_mat.astype(int)
    adj_mat_order = np.zeros([len(adj_mat), 6])
            
    for i in range(len(adj_mat)):
        neighs = ico_6_coords[adj_mat[i,:], :]
    
        center_pt = ico_6_coords[i]
        neighs_angle = compute_angles(neighs, center_pt)
    
    
        neighs_angle = neighs_angle + np.pi/4;
        neighs_angle = np.mod(neighs_angle, 2*np.pi)
        args = np.argsort(neighs_angle);
        adj_mat_order[i,:] = adj_mat[i,args]

    adj_mat_intermediate = adj_mat
    level = levels[n]
    
    np.save('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices' + '/adj_mat_order_' + str(level) + '.npy', adj_mat_order)

print('complete')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:23:37 2020

@author: fa19
"""
import numpy as np
import scipy.io as sio
def ismember(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = np.isin(a_vec,b_vec)
    common = a[bool_ind]
    common_unique, common_inv  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]



def compute_angles(points, centre):
    neighs_projected = np.zeros_like(points)
    
    for j in range(len(points)):
        t = 1 - (np.matmul(centre,np.transpose(points[j])))/(centre[0]**2 + centre[1]**2 + centre[2]**2)
        neighs_projected[j] = [points[j,0] + (centre[0]*t), points[j,1] + (centre[1]*t), points[j,2] + (centre[2]*t)]

    neighbour_angles = np.zeros(len(points))
    
    if centre[0] != 0 or centre[1] != 0:
        n_x = np.cross([0,0,1], centre)
        n_x = n_x/np.linalg.norm(n_x) #normal of the new projected x-axis
        n_y = np.cross(centre, n_x) # normal of the new projected y-axis
        n_y = n_y / np.linalg.norm(n_y)
    else:
        n_x = [1,0,0]
        n_y = [0,1,0]

    assert abs(np.matmul(n_x, np.transpose(n_y))) < 1e-6
    assert any(n_x != np.zeros(len(n_x)))==True #n_x * n_y != 0');
    for j in range(len(points)):
        # calculate the angle between neighs and x-axis
        line_between = neighs_projected[j] - centre
        
        
        tmp = np.matmul(line_between, np.transpose(n_x))
        tmp = tmp/np.linalg.norm(n_x)
        tmp = tmp / np.linalg.norm(line_between) 

        if tmp > 1.0:
            tmp = 1.0
        elif tmp < -1.0:
            tmp = -1.0

        neighbour_angles[j] =  np.arccos(tmp);
        if np.matmul((neighs_projected[j] - centre) , np.transpose(n_y)) < 0:
            neighbour_angles[j] = 2*np.pi - neighbour_angles[j]
            
    return neighbour_angles


def my_Get_neighs_order():
    neigh_orders_40962 = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_6.npy')
    neigh_orders_10242 = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_5.npy')
    neigh_orders_2562 = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_4.npy')
    neigh_orders_642 = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_3.npy')
    neigh_orders_162 = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_2.npy')
    neigh_orders_42 = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_1.npy')
    neigh_orders_12 = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_0.npy')
    
    return neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12
  
def my_get_neighs_order(order_path):
    adj_mat_order = np.load(order_path)
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders



def my_mat_Get_neighs_order():
    neigh_orders_40962 = my_mat_get_neighs_order('neighbour_indices_matlab/adj_mat_order_40962.mat')
    neigh_orders_10242 = my_mat_get_neighs_order('neighbour_indices_matlab/adj_mat_order_10242.mat')
    neigh_orders_2562 = my_mat_get_neighs_order('neighbour_indices_matlab/adj_mat_order_2562.mat')
    neigh_orders_642 = my_mat_get_neighs_order('neighbour_indices_matlab/adj_mat_order_642.mat')
    neigh_orders_162 = my_mat_get_neighs_order('neighbour_indices_matlab/adj_mat_order_162.mat')
    neigh_orders_42 = my_mat_get_neighs_order('neighbour_indices_matlab/adj_mat_order_42.mat')
    neigh_orders_12 = my_mat_get_neighs_order('neighbour_indices_matlab/adj_mat_order_12.mat')
    
    return neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12
  
def my_mat_get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders


def my_mat_Get_2ring_neighs_order():
    neigh_orders_2ring_40962 = my_mat_get_2ring_neighs_order('neighbour_indices_matlab/adj_mat_order_2ring_40962.mat')
    neigh_orders_2ring_10242 = my_mat_get_2ring_neighs_order('neighbour_indices_matlab/adj_mat_order_2ring_10242.mat')
    neigh_orders_2ring_2562 = my_mat_get_2ring_neighs_order('neighbour_indices_matlab/adj_mat_order_2ring_2562.mat')
    neigh_orders_2ring_642 = my_mat_get_2ring_neighs_order('neighbour_indices_matlab/adj_mat_order_2ring_642.mat')
    neigh_orders_2ring_162 = my_mat_get_2ring_neighs_order('neighbour_indices_matlab/adj_mat_order_2ring_162.mat')
    neigh_orders_2ring_42 = my_mat_get_2ring_neighs_order('neighbour_indices_matlab/adj_mat_order_2ring_42.mat')
    
    return neigh_orders_2ring_40962, neigh_orders_2ring_10242, neigh_orders_2ring_2562, neigh_orders_2ring_642, neigh_orders_2ring_162, neigh_orders_2ring_42

def my_mat_get_2ring_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order_2ring']
    neigh_orders = np.zeros((len(adj_mat_order), 19))
    neigh_orders[:,0:18] = adj_mat_order-1
    neigh_orders[:,18] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders




def my_mat_Get_2ring_upconv_index():
    
    upconv_top_index_40962, upconv_down_index_40962 = my_mat_get_2ring_upconv_index('neighbour_indices_matlab/adj_mat_order_40962.mat')
    upconv_top_index_10242, upconv_down_index_10242 = my_mat_get_2ring_upconv_index('neighbour_indices_matlab/adj_mat_order_10242.mat')
    upconv_top_index_2562, upconv_down_index_2562 = my_mat_get_2ring_upconv_index('neighbour_indices_matlab/adj_mat_order_2562.mat')
    upconv_top_index_642, upconv_down_index_642 = my_mat_get_2ring_upconv_index('neighbour_indices_matlab/adj_mat_order_642.mat')
    upconv_top_index_162, upconv_down_index_162 = my_mat_get_2ring_upconv_index('neighbour_indices_matlab/adj_mat_order_162.mat')
    upconv_top_index_42, upconv_down_index_42 = my_mat_get_2ring_upconv_index('neighbour_indices_matlab/adj_mat_order_42.mat')
    upconv_top_index_12, upconv_down_index_12 = my_mat_get_2ring_upconv_index('neighbour_indices_matlab/adj_mat_order_12.mat')
    return  upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42,upconv_top_index_12, upconv_down_index_12


def my_mat_get_2ring_upconv_index(order_path):  
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index



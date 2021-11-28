#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:23:44 2020

@author: fa19
"""

import scipy.io as sio

adj_6 = sio.loadmat('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices_matlab/adj_mat_order_2ring_40962.mat')['adj_mat_order_2ring']-1
adj_5 = sio.loadmat('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices_matlab/adj_mat_order_2ring_10242.mat')['adj_mat_order_2ring']-1
adj_4 = sio.loadmat('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices_matlab/adj_mat_order_2ring_2562.mat')['adj_mat_order_2ring']-1
adj_3 = sio.loadmat('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices_matlab/adj_mat_order_2ring_642.mat')['adj_mat_order_2ring']-1
adj_2 = sio.loadmat('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices_matlab/adj_mat_order_2ring_162.mat')['adj_mat_order_2ring']-1
adj_1 = sio.loadmat('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices_matlab/adj_mat_order_2ring_42.mat')['adj_mat_order_2ring']-1



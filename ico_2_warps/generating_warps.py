#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:52:55 2020

@author: fa19
"""




import numpy as np



import nibabel as nb

ico_3 = nb.load('/home/fa19/Downloads/icosahedrons/ico-2.surf.gii')

coords = ico_3.darrays[0].data / 100


def find_distance(arr1, arr2):
    return np.arccos(np.dot(arr1, arr2))

minimum = 1000
for row in coords:
    for row2 in coords:
        if list(row)!=list(row2):
            d = find_distance(row, row2)
                
            if d < minimum:
                if d !=0:
                    
                    minimum = d
                
                
smallest_distance = minimum

#l = smallest_distance/3

def gen_new_coord(l):
    sol = np.zeros(3)
    
    x = np.random.uniform(-l,l)
    y_max = np.sqrt(l-(x**2))
    
    y = np.random.uniform(-y_max, y_max)
    
    sol[0] = x
    sol[1] = y
    sol[2] = np.sqrt(1- (x**2 + y**2))
    return sol





def my_rodrigues_rot(current_point, target = [0,0,1]):
    
    """
    Input: an input UNIT vector and a target UNIT vector (default == [0, 0, 1]  i.e the north pole)
    
    Output: the 3x3 rotation matrix that takes the input vector to the target vector. 
            Calculated via the Rodrigues Formula Method
    
    """
    
    
    current_point = current_point/np.linalg.norm(current_point)
    
    target = target / np.linalg.norm(target)
    
    
    
    if type(current_point) == np.ndarray or type(target) == np.ndarray:
        assert any(current_point != target) == True
    elif type(current_point) == list and type(target) == list:
        assert current_point != target
    else:
        raise TypeError('target or current point have strange typing. Neither list nor numpy array')
        
    
    
    angle = np.arccos(np.dot(current_point, target))
    
    unit_vector_between_them = np.cross(current_point, target)/np.sin(angle)
    unit_vector_between_them = unit_vector_between_them/np.linalg.norm(unit_vector_between_them)
    
   
    kx = unit_vector_between_them[0]
    ky = unit_vector_between_them[1]
    kz = unit_vector_between_them[2]
    
    K = np.array([[0,-kz, ky],[kz, 0, -kx], [-ky, kx,0]])
    
    K_squared = np.linalg.matrix_power(K,2)
    
    ROT = np.identity(3) + np.sin(angle) * K + (1-np.cos(angle))*K_squared

    return ROT

def z_to_point(arr):
    return np.linalg.inv(my_rodrigues_rot(arr))

#sol = np.zeros_like(coords)
    
l = minimum / 8

for k in range(100):
    ico_3 = nb.load('/home/fa19/Downloads/icosahedrons/ico-2.surf.gii')

    coords = ico_3.darrays[0].data / 100
    counter = 0
    for row in coords:
        failed = True
        while failed == True:
            adjustment = np.random.uniform(-l,l, size = 3)
            
            new_row =  row + adjustment
            p = find_distance(row, new_row)
            if p <= l:
                failed = False
                coords[counter] = new_row
                counter+=1
    
    
    
    
    
    
    ico_3.darrays[0].data = coords * 100
    nb.save(ico_3, '/home/fa19/Documents/Benchmarking/ico_2_warps/ico_2_'+str(k)+'.surf.gii')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:27:34 2021

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:53:36 2020

@author: fa19
"""



import nibabel as nb
import numpy as np

all_coords = []


def make_polar(array_of_cartesian_coordinates):
    x,y,z = array_of_cartesian_coordinates[:,0], array_of_cartesian_coordinates[:,1], array_of_cartesian_coordinates[:,2]
    r2 = np.square(x) + np.square(y) + np.square(z)

    r = np.sqrt(r2)
    phi = np.arctan2(y,x)
    theta = np.arccos(z,np.sqrt(r2))

    return phi, theta, r



import math

def angle_between(P, Q):
    #The angle between points P and Q
    
    dot_product = np.dot(P, Q)
    
    norm1 = np.linalg.norm(P)
    norm2 = np.linalg.norm(Q)
    norm = norm1 * norm2
    return np.arccos(dot_product / norm)



     

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
    if angle != np.pi:
        unit_vector_between_them = np.cross(current_point, target)/np.sin(angle)
        unit_vector_between_them = unit_vector_between_them/np.linalg.norm(unit_vector_between_them)
    else:
        unit_vector_between_them = np.array([1,0,0])
        
    kx = unit_vector_between_them[0]
    ky = unit_vector_between_them[1]
    kz = unit_vector_between_them[2]
    
    K = np.array([[0,-kz, ky],[kz, 0, -kx], [-ky, kx,0]])
    
    K_squared = np.linalg.matrix_power(K,2)
    
    ROT = np.identity(3) + np.sin(angle) * K + (1-np.cos(angle))*K_squared

    return ROT


def axis_rot(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotation_matrix(f,t):
# a and b are in the form of numpy array
    v = np.cross(f, t)
    u = v/np.linalg.norm(v)
    c = np.dot(f, t)
    h = (1 - c)/(1 - c**2)
    
    vx, vy, vz = v
    rot =[[c + h*vx**2, h*vx*vy - vz, h*vx*vz + vy],
          [h*vx*vy+vz, c+h*vy**2, h*vy*vz-vx],
          [h*vx*vz - vy, h*vy*vz + vx, c+h*vz**2]]
   
    return rot

for k in range(7):
    
     
    icosahedron_coords = nb.load('/home/fa19/Downloads/icosahedrons/ico-'+str(k)+'.surf.gii').darrays[0].data
    
    
    all_coords.append(icosahedron_coords/100)
    


original_coords = all_coords[6]

start_point = 0

ROT = my_rodrigues_rot(original_coords[start_point])

original_coords = np.matmul(ROT, original_coords.T).T


original_coords = np.divide(original_coords.T , np.linalg.norm(original_coords, axis=1) ).T
#original_coords = np.round(original_coords, 9)


original_faces = nb.load('/home/fa19/Downloads/icosahedrons/ico-'+str(0)+'.surf.gii').darrays[1].data


reference_point = original_coords[0] #this is the chosen reference SHOULD BE 100


middle = np.zeros(3)
print(reference_point , ' - this should be 001')

neighbour_point = original_coords[3] #this is an arbitrary choice of neighbour, just cos 3 is a neghbour of 0
middle[2] = neighbour_point[2]


mapping_array = np.zeros([60,40962])


def get_mapping_dict(newCoords, oldCoords):
    mapping_dict = {}
    for i in range(len(oldCoords)):
        original_vector = np.round(oldCoords[i],11)
        
        differences = np.linalg.norm(newCoords - original_vector, axis=1)
        location = np.argmin(differences)
        
        mapping_dict[i] = location
    return mapping_dict

def record_orientation(mapping_dict,mapping_array, count):
    orientation = list(mapping_dict.values())
    print(len(set(orientation)))
    mapping_array[count] = orientation
    count += 1
    
    return count, mapping_array

def find_random_neighbour_point(P, original_faces):
    for row in original_faces:
        if P in row:
            if row[0]!=P:
                return row[0]
            else:
                return row[1]
            
    else:
        print('Error not found')
        raise ValueError

    
    
    
count  = 0
for j in range(12): #for each vertex
    if j == start_point:
        new_coords = original_coords
        
        
    
    elif j != start_point: #if you arent the reference point
        
        #First take the vertex to the reference vertex
        
        new_point = original_coords[j]
        
        
        
        ROT = my_rodrigues_rot(original_coords[j], reference_point)
        new_coords = np.matmul(ROT , original_coords.T).T
        
#        new_coords = np.round(new_coords, 9)
        
        new_neighbourhood_point = new_coords[find_random_neighbour_point(j, original_faces)]

        angle = angle_between(neighbour_point - middle, new_neighbourhood_point-middle)
        
        
        
        
        
        rot = axis_rot([0,0,1], angle)
        
        new_coords = np.matmul(rot, new_coords.T).T
        


    for r in range(5):
        if r !=0:
        
            
            size_of_rot = r
    
    
            ROT = axis_rot(reference_point,r*2*np.pi/5)
        
            final_coords = np.matmul(ROT, new_coords.T).T
        else:
            final_coords = new_coords
            
            
        mapping_dict = {}
        
    
        mapping_dict = get_mapping_dict(final_coords, original_coords)
        count, mapping_array = record_orientation(mapping_dict, mapping_array,count)
    
    
            
                        
                        
#        orientation = list(mapping_dict.values())
#        mapping_array[count] = orientation
#        count+=1
        print('Completed Rotation Number ', str(count))
         
        # TESTING ORIENTATION
        
        
#        
#        file_1 = nb.load('/home/fa19/Documents/dHCP_Data/Raw/curvature/CC00050XX01-7201-curvature.shape.gii')
#        nb.save(file_1,'/home/fa19/Documents/my_version_spherical_unet/testing_rotations/original.shape.gii')
#        
#        
#        file_1.darrays[0].data = file_1.darrays[0].data[orientation]
#        nb.save(file_1,'/home/fa19/Documents/my_version_spherical_unet/testing_rotations/rotated.shape.gii')

         
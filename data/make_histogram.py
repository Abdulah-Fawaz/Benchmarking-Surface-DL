#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:50:18 2021

@author: fa19
"""


import numpy as np

import pandas as pd


    
df = pd.read_excel('/home/fa19/Downloads/dHCP_metadata_release3.xlsx') # requires installation of odfpy (use pip or conda)



#### ############

# CHOOSE DATA ARRRRAY HERE
ds_array = np.load('bayley/full.npy', allow_pickle=True)


############
ds_array_ids = np.zeros(len(ds_array))

total_number = len(ds_array)


prems_count = 0
males_count = 0

for i in range(len(ds_array_ids)):
    session = ds_array[i,0].split('_')[-1]
    ds_array_ids[i] = session
    individual = df[df['Scan validation ID']==int(session)]
    
    sex = individual['Sex (1=male; 2=female)'].item()
    
    if sex == 1:
        males_count+=1
    elif sex !=2:
        print(sex)
        raise ValueError
    birth_age   = individual['GA at birth'].item()
    if birth_age <37:
        prems_count+=1
    


print('total is ',total_number)
print('prems are ', prems_count)
print('terms are ', total_number - prems_count)
print('males are ', males_count)
print('females arrre', total_number - males_count)
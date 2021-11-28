#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:24:05 2021

@author: fa19
"""

import numpy as np

import os
import copy
root_dir = '/data/abdulah/'

for file in os.listdir(root_dir):
    if 'sub-' in file:
    
        new_file = file.replace('sub-', '')
        new_file = new_file.replace('ses-','')
         
        os.rename(root_dir + file, root_dir+new_file)
        
        
for file in os.listdir(root_dir):
    if '_W' in file:
        before, after = file.split('_W')
        before=  before + '_W'
        number = after.split('.shape.gii')[0]
        number = int(number)
        if number == 0:
            
            after = '.shape.gii'
            number = str(100)
        
            new_file = before + number + after
            os.rename(root_dir + file, root_dir + new_file)
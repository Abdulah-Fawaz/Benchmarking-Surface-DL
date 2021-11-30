#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:10:37 2020

@author: fa19
"""

import numpy as np

import pandas as pd


    
df = pd.read_excel('dHCP_third_release.ods', engine = 'odf') # requires installation of odfpy (use pip or conda)






def make_into_filename(subject, session):
    #file convention combines subject and sesssion into format: sub-SUBJECT_ses-SESSION
    return 'sub-' + str(subject) + '_ses-' + str(session)


"""

scan age filtering:

filter any scan such that birth age is <37 but scan age >=37.

"""

scan_age_filtered = df.loc[np.logical_not(np.logical_and((df[df.columns[3]]<37) ,#preterm  
                           (df[df.columns[4]]>=37))) ] # scan
                



"""

birth age filtering:
    
filter any scan such that scan age and birth ages are <37)


"""

birth_age_filtered = df[np.logical_not(
        np.logical_and(
                df[df.columns[3]]<37 , #preterm
                df[df.columns[4]]<37 # scan age > 37
                ))
        ]


original_array = scan_age_filtered
d = np.array(original_array)
d = d[:,3:5]

#d[:,1] = np.sort(d[:,1])

colors = ['red', 'blue']
p = d[:,0]<37

C = [colors[item] for item in p]

#plt.scatter(d[:,1],np.arange(len(d)), color = C, s = 5)

plt.scatter(np.round(2*d[:,1].astype(float), 0)/2,np.arange(len(d)), color = C, s = 5)


plt.savefig('fig_saved')
plt.close()



all_unique, all_counts = np.unique(np.round(2*d[:,1].astype(float))/2, return_counts = True)

preterms = d[d[:,0]<37]
preterms_scan_ga = preterms[:,1]
preterms_unique, preterm_counts = np.unique( np.round(2 * preterms[:,1].astype(float))/2, return_counts= True)


p_counts = []
for item in all_unique:
    for i in range(len(preterms_unique)):
        found = 0
        if preterms_unique[i] == item:
            p_counts.append(preterm_counts[i])
            found = 1
            break
    if found == 0:
        p_counts.append(0)


t_counts = all_counts - np.array(p_counts)


# the x locations for the groups
barWidth = 0.35       # the width of the bars: can also be len(x) sequence



bars1 = t_counts
bars2 = p_counts
ticks = []
for item in r:
    if item%1 == 0:
        ticks.append(item)
        

# The position of the bars on the x-axis
r = all_unique
# Names of group and bar width
barWidth = 0.5
names = all_unique
# Create brown bars
bar1 = plt.bar(r, bars1, color='dodgerblue', edgecolor='white', width=barWidth)
# Create green bars (middle), on top of the firs ones
bar2 = plt.bar(r, bars2, bottom=bars1, color='orange', edgecolor='white', width=barWidth)

inds =  np.arange(r.min()//1,1+r.max()//1, 1)

plt.xticks( inds, inds.astype(int))#, fontweight='bold')
plt.legend((bar1, bar2), ('Term', 'Preterm'))
plt.xlabel("Scan Age")
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Scan Ages')
        

plt.savefig('Bar_Graph_for_Scan_Age_Prediction')

#plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
#plt.yticks(np.arange(0, 81, 10))
#plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()







def make_into_filename(subject, session):
    #file convention combines subject and sesssion into format: sub-SUBJECT_ses-SESSION
    return 'sub-' + str(subject) + '_ses-' + str(session)


"""

make dataset array

"""
dataset_arr = np.empty([len(scan_age_filtered),3], dtype = object)
counter = 0
for row in np.array(scan_age_filtered):
    
    
    filename = make_into_filename(row[0], row[2])
    
    scan_age = row[3:5] # row 4 for scan age!
    dataset_arr[counter,0] = filename
    dataset_arr[counter,1:] = scan_age
    counter = counter + 1
    
    



preterms = dataset_arr[dataset_arr[:,1]<37]
    
terms = dataset_arr[dataset_arr[:,1]>=37]
    
    


training_ratio = 0.8 

test_ratio = 0.1

#validation_ratio = 0.1

training_terms, validation_terms, test_terms = a,b,c = np.split(terms, [int(len(terms)*training_ratio), int(len(terms)*(training_ratio+ test_ratio))])

training_preterms, validation_preterms, test_preterms = a,b,c = np.split(preterms, [int(len(preterms)*training_ratio), int(len(preterms)*(training_ratio+ test_ratio))])


train_set = np.vstack([training_terms, training_preterms])

validation_set = np.vstack([validation_terms, validation_preterms])

test_set = np.vstack([test_terms, test_preterms])


train_set = train_set[:,[0,2]]
validation_set = validation_set[:, [0,2]]
test_set = test_set[:,[0,2]]

np.save('train_scan_ga', train_set)
np.save('validation_scan_ga', validation_set)
np.save('test_scan_ga', test_set)



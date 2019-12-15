#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 00:04:40 2019

@author: saransh
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def missingValues(data):
    missing_values = data.isna().sum()
    print(missing_values)
    print("Total missng values: ",missing_values.sum())


data = pd.read_csv("winequality-red_miss.csv")

Series_row_count_na = data.isna().sum(axis = 1)
list_miss = Series_row_count_na.tolist()
count = Counter(list_miss)
print(count)

lists = sorted(count.items())
x,y = zip(*lists)
plt.plot(x,y)
plt.show()



for key,value in count.items():
    if key >=6:
        print(key,value)
        
droped = []

for i in range(len(data.index)):
    if list_miss[i]>=6:
        droped.append(i)
    
print("index of rows dropped:\n", droped)    
new_droped = []
new_data = data.drop(droped)


lis = new_data["quality"].isna()
for i in [x for x in range(len(lis)) if x not in droped]:
    if lis[i]:
        new_droped.append(i)

new_new_data = new_data.drop(new_droped)
missingValues(new_new_data)
      
#print(droped)
        
        
        
        

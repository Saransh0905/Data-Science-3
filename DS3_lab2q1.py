#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:03:00 2019

@author: saransh
"""
import pandas as pd
import numpy as np

def missingValues(data):
    missing_values = data.isna().sum()
    print("\n\n",missing_values)
    print("Total missing values: ",missing_values.sum())

data = pd.read_csv("winequality-red_miss _COPY.csv")
missingValues(data)

flag2 = 0
flag1= 0
i = 0
while flag1<2 and flag2<2:
    if data.iloc[i, data.columns.get_loc("fixed acidity")] != np.nan:
        data.iloc[i, data.columns.get_loc("fixed acidity")] = np.nan
        flag1+=1
    if data.iloc[i, data.columns.get_loc("volatile acidity")] != np.nan:
        data.iloc[i, data.columns.get_loc("volatile acidity")] = np.nan
        flag2+=1
    i+=1
    
missingValues(data)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:30:58 2019

@author: saransh
"""
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import statistics as st


miss = pd.read_csv("winequality-red_miss.csv")
data = pd.read_csv("winequality-red_original.csv",sep = ';')
miss_median=miss.fillna(miss.median())
attributes = data.columns
print("________________mean_data______________________")
print(data.mean())
print("________________mean_median_data______________________")
print(miss_median.mean())
print("_______________median_data_________________")
print(data.median())
print("_______________median_median_data_________________")
print(miss_median.median())
print("_______________mode_data___________________")
print(data.mode())
print("_______________mode_median_data_________________")
print(miss_median.mode())
print("________________std_data___________________")
print(data.std())
print("_______________std_median_data_________________")
print(miss_median.std())
print("_______________________________________")

col=attributes[2:]

for i in col:
    print(i)
    plt.boxplot([list(map(float,data[i])),list(map(float,miss_median[i]))])
    plt.show()
    print("_______________________________________")

rms_median=data[col]-miss_median[col]
print(((rms_median**2).mean())**(0.5))



miss_interpolate=miss.interpolate()

print(data.mean())
print(miss_interpolate.mean())
print("__________________median_____________________")
print(data.median())
print(miss_interpolate.median())
print("__________________mode_____________________")
print(data.mode())
print(miss_interpolate.mode())
print("__________________std_____________________")
print(data.std())
print(miss_interpolate.std())
print("_______________________________________")

for i in col:
    print(i)
    plt.boxplot([list(map(float,data[i])),list(map(float,miss_interpolate[i]))])
    plt.show()
    print("_______________________________________")

rms_interpolate=data[col]-miss_interpolate[col]
print("interpolate")
interpolate = ((rms_interpolate**2).mean())**(0.5)
print(interpolate)
print("_______________________________________")

print("median")
rms_median=data[col]-miss_median[col]
print(((rms_median**2).mean())**(0.5))

miss_0=miss.fillna(0)
plt.hist(miss_0['chlorides'])
plt.show()
plt.hist(miss_median['chlorides'])
plt.show()
plt.hist(miss_interpolate['chlorides'])
plt.show()


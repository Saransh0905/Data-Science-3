#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:35:41 2019

@author: saransh
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.mixture import GaussianMixture


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


data = pd.read_csv("SteelPlateFaults-2class.csv")
cols = data.columns
#data0 = data[data['Z_Scratch']==0]
#data1 = data[data['Z_Scratch']==1]
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42,shuffle = True)
new = pd.concat([X_train,Y_train],axis = 1)
X_train0 = new[new["Z_Scratch"]==0].iloc[:,:-1]
X_train1 = new[new["Z_Scratch"]==1].iloc[:,:-1]
gmm0 = GaussianMixture(n_components=1).fit(X_train0)
gmm1 = GaussianMixture(n_components=1).fit(X_train1)

Y_pred = []
zero = gmm0.score_samples(X_test)
one= gmm1.score_samples(X_test)
for i in range(len(X_test)):
     if zero[i]<one[i]:
         Y_pred.append(1)
     else:
         Y_pred.append(0)


'''
scale_minmax = MinMaxScaler()

X_minmax = scale_minmax.fit_transform(X)
scale_std = StandardScaler()
scale_std.fit(X)
print(scale_std.transform(X)) 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42,shuffle = True)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
m = confusion_matrix(Y_test,Y_pred)
print(accuracy_score(Y_test,Y_pred))
'''
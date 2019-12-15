#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:36:21 2019

@author: saransh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

def KNN(x_train,y_train,x_test):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

def Divide(x,y):
    Array1=[]
    Array0=[]
    for i in range(n):
        if y_train[i]==0:
            Array0.append(x_pca_train[i])
        else:
            Array1.append(x_pca_train[i])
    Array1=np.array(Array1)
    Array0=np.array(Array0)
    return Array0,Array1



data=pd.read_csv("pima-indians-diabetes.csv")
col=data.columns
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42,shuffle=True)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
red=[]
knnacc=[]
bayacc=[]
n=len(y_train)
nt=len(y_test)
for j in range(1,27):
    pca = PCA(n_components=j)
    red.append(j)
    pca.fit(X_train)
    x_pca_train = pca.transform(X_train)
    x_pca_test = pca.transform(X_test)
    a=[]
    b=[]
    c=[]
    d=[]
    print("___________________________")
    print("Reduced dimensions to l =",j)
    print("  ")
    l = []
    
    for i in range(1,22,2):
        y_pred = KNN(x_pca_train,y_train,x_pca_test)
        c.append(confusion_matrix(y_test, y_pred))
        a.append(accuracy_score(y_test, y_pred))
        b.append(i)
    plt.plot(b,a)
    plt.show()
    print("KNN-Value of k for which accuracy(knn algorithm) is maximum -",2*a.index(max(a))+1,"and Accuracy is",max(a))

    
    Array0,Array1=Divide(x_pca_train,y_train)
    
    muA0=[]
    muA1=[]
    
    for i in range(j):
        muA0.append(np.mean(Array0[i]))
        muA1.append(np.mean(Array0[i]))
    tp=0
    tn=0
    Array1=Array1.transpose()
    cov1=np.cov(Array1)
    Array0=Array0.transpose()
    cov0=np.cov(Array0)
    for i in range(nt):
        p1=multivariate_normal.pdf(x_pca_test[i], mean=muA1, cov=cov1,allow_singular=True)
        p0=multivariate_normal.pdf(x_pca_test[i], mean=muA0, cov=cov0,allow_singular=True)
        y_pred = [] 
        if p1>p0:
            if y_test[i]==1:
                tp+=1
        else:
            if y_test[i]==0:
                tn+=1
    
    print('The accuracy % by bayes is:',100*(tp+tn)/nt)

    print("___________________________")

    knnacc.append(max(a))
    bayacc.append((tp+tn)/nt)

print("___________________________")
print("Plot for KNN accuracy vs no. of dimensions")
plt.plot(red,knnacc)
plt.ylim((0,1))
plt.xlabel("No. of reduced dimensions")
plt.ylabel("KNN accuracy")
plt.show()
print("___________________________")
print("Plot for Bayes accuracy vs no. of dimensions")
plt.plot(red,bayacc)
plt.ylim((0,1))
plt.xlabel("No. of reduced dimensions")
plt.ylabel("Bayes accuracy")
plt.show()
print("Maximum accuracy for KNN after reducing dimensions is",max(knnacc),"for number of dimesions =",red[knnacc.index(max(knnacc))])
print("Maximum accuracy for Bayes after reducing dimensions is",max(bayacc),"for number of dimesions =",red[bayacc.index(max(bayacc))])
print("___________________________")
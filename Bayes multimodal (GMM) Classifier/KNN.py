#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:48:17 2019

@author: saransh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data=pd.read_csv("SteelPlateFaults-2class.csv")
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
for j in range(1,len(col)):
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
    for i in range(1,22,2):
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(x_pca_train, y_train)
        y_pred = classifier.predict(x_pca_test)
        c.append(confusion_matrix(y_test, y_pred))
        a.append(accuracy_score(y_test, y_pred))
        b.append(i)
        d.append([accuracy_score(y_test, y_pred),i])
    print("Value of k for which accuracy(knn algorithm) is maximum(knn algorithm) -",max(d)[1],"and the Accuracy is",max(d)[0])
    
    classifier = GaussianNB()
    classifier.fit(x_pca_train, y_train)
    y_pred = classifier.predict(x_pca_test)
    cm = confusion_matrix(y_test, y_pred)
    print("The confusion matrix for bayes classifier is",cm)
    print("Accuracy for Baye's classification -",accuracy_score(y_test, y_pred))
    print("___________________________")
    
    knnacc.append(max(d)[0])
    bayacc.append(accuracy_score(y_test, y_pred))

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
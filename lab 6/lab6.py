#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:03:41 2019

@author: saransh
"""

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def KNN(x_train,y_train,x_test):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

def Bayes(x_train,y_train,x_test):
    classifier = GaussianNB()
    classifier.fit(x_pca_train, y_train)
    y_pred = classifier.predict(x_pca_test)
    return y_pred
    

def divide(x_train,y_train,col):
    df0 = []
    df1 = []
    for i in range(len(x_train)):
        #di = makedict(col[:-1],x_train[i])
        #print(di)
        if y_train[i]==1:
            x = list(x_train[i])
            x.append(1)
            df1.append(x)
        else:
            x = list(x_train[i])
            x.append(0)
            df0.append(x)
    return df0,df1

def makedict(key,value):    
    di = dict()
    for i in range(len(key)):
        di[key[i]]=value[i]
    print(di)
    return pd.DataFrame(di,index = [0])
def calMean(df):
    m = 0
    for i in range(len(df)):
        ar = np.array(df[i])
        m += ar.mean()
    
 
data=pd.read_csv("pima-indians-diabetes.csv")
col=data.columns
#plt.hist(data[col[2]],bins = 100)
#print(data[col[-1]])
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42,shuffle=True)
df0,df1 = divide(X_train,y_train,col)
#scaler = StandardScaler()
#scaler.fit(X_train)

#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


        
        

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
    print("___________________________")
    print("Reduced dimensions to l =",j)
    print("  ")
    for i in range(1,22,2):
        y_pred = KNN(x_pca_train,y_train,x_pca_test)
        c.append(confusion_matrix(y_test, y_pred))
        a.append(accuracy_score(y_test, y_pred))
        b.append(i)
    print("KNN-Value of k for which accuracy(knn algorithm) is maximum -",2*a.index(max(a))+1,"and Accuracy is",max(a))
    
    y_pred = Bayes(x_pca_train,y_train,x_pca_test)
    cm = confusion_matrix(y_test, y_pred)
    print("The confusion matrix for bayes classifier is",cm)
    print("Accuracy for Baye's classification -",accuracy_score(y_test, y_pred))
    print("___________________________")
    
    knnacc.append(max(a))
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
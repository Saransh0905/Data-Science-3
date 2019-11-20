#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:56:58 2019

@author: saransh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


csv = pd.read_csv("Iris.csv")
datay = csv["Species"]
data = csv.drop(columns = ["Species"],axis = 1)
#pca = PCA(n_components=2).fit(data)
data = PCA(n_components=2).fit_transform(data)
data=pd.DataFrame(data)
print("_________________")
print(" ")
print("2D-Points After Reducing Dimensions ")
print("_________________")
plt.scatter(data.iloc[:,0],data.iloc[:,1],color="blue")
plt.show()


Kmean = KMeans(n_clusters=3)
labels = Kmean.fit_predict(data)
#labels = Kmean.predict(data)
print("_________________")
print(" ")
print("KMeans Clustering")
print("_________________")
plt.scatter(data.iloc[:,0],data.iloc[:,1],c=Kmean.labels_,cmap='viridis')
centers=Kmean.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],s=100,c='black')
plt.show()
print("purity score for KMeans Clustering is -")
print(purity_score(pd.DataFrame(datay),pd.DataFrame(labels)))
print("_________________")

print(" ")
print(" ")
print(" ")

print("_________________")
print(" ")
print("Agglomerative Clustering")
print("_________________")
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
labelag=cluster.fit_predict(data)
plt.scatter(data.iloc[:,0],data.iloc[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
'''
Kmean.fit(data)
labels = Kmean.predict(data)
print("_________________")
print(" ")
print("KMeans Clustering")
print("_________________")
plt.scatter(data.iloc[:,0],data.iloc[:,1],c=labels,cmap='viridis')
centers=Kmean.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],s=100,c='black')
plt.show()
print("purity score for KMeans Clustering is -")
print(purity_score(pd.DataFrame(datay),pd.DataFrame(labels)))
print("_________________")

plt.show()
'''
print("purity score for Agglomerative Clustering is -")
print(purity_score(pd.DataFrame(datay),pd.DataFrame(labelag)))
print("_________________")

print(" ")

print(" ")

print("_________________")
print(" ")
print("DBSCAN")
print("_________________")

epsp=[0.05,0.5,0.95]
min_samplesp=[1,5,10,20]
ps=[]
arr = []
for i in epsp:
    for j in min_samplesp:
        db = DBSCAN(eps = i, min_samples = j)
        arr.append([i,j])
        labels1 = db.fit_predict(data)
        ps.append([purity_score(pd.DataFrame(datay),pd.DataFrame(labels1)),i,j])
psmax=max(ps)
ind = ps.index(psmax)
print('for eps = 0.05 and minpts = 1')
db = DBSCAN(eps = 0.05, min_samples = 1).fit(data) 
labels1 = db.labels_
plt.scatter(data.iloc[:,0],data.iloc[:,1], c=db.labels_)
plt.show()

db = DBSCAN(eps = arr[ind][0], min_samples = arr[ind][1]).fit(data) 
labels1 = db.labels_
plt.scatter(data.iloc[:,0],data.iloc[:,1], c=db.labels_, cmap='rainbow')
plt.show()

print("purity score for DBSAN is -")
print(purity_score(pd.DataFrame(datay),pd.DataFrame(labels1)))
print("_________________")
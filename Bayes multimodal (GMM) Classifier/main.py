# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 23:12:05 2019
@author: Saransh
"""

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def train_test_split(data,test_size,seed): 
    return model_selection.train_test_split(data.iloc[:,0:-1],data.iloc[:,-1],test_size=test_size,random_state=seed) 

def standardisation(dataframe): 
    return StandardScaler().fit_transform(dataframe.values)

def stdatts(dataframe): 
    dataframe.iloc[:,0:-1] = standardisation(dataframe.iloc[:,0:-1])

def makegmm(c1,c2,q): 
    return GaussianMixture(n_components=q,covariance_type="tied").fit(c1),GaussianMixture(n_components=q,covariance_type="tied").fit(c2)

def classifieddata(data_train,c1,c2): 
    return data_train[data_train["Z_Scratch"]==c1].iloc[:,0:-1],data_train[data_train["Z_Scratch"]==c2].iloc[:,0:-1]


data = pd.read_csv("SteelPlateFaults-2class.csv")
stdatts(data)

q_values = [1,2,4,8,16]

X_train, X_test, Y_train, Y_test = train_test_split(data,0.3,42)
X_train0,X_train1=classifieddata(pd.concat([X_train,Y_train],axis=1),0,1)
print("Original data:")
for q in q_values:
    gmm0,gmm1 = makegmm(X_train0,X_train1,q)
    
    Y_pred=[int(gmm0.score_samples(X_test)[i]<gmm1.score_samples(X_test)[i]) for i in range(len(X_test))]
    
    print([list(confusion_matrix(Y_test,Y_pred)[i]) for i in range(len(confusion_matrix(Y_test,Y_pred)))],'\t',accuracy_score(Y_test,Y_pred))


for i in range(1,len(list(data))):
    pca = PCA(n_components=i)
    
    reduced_data = pd.concat([pd.DataFrame(data = pca.fit_transform(data.copy().iloc[:,0:-1]),columns=['comp '+str(n) for n in range(i)]),data.copy().iloc[:,-1]], axis = 1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(reduced_data,0.3,42)
    X_train0,X_train1=classifieddata(pd.concat([X_train,Y_train],axis=1),0,1)
    
    print("\nn_dimensions="+str(i))
    for q in q_values:
        gmm0,gmm1 = makegmm(X_train0,X_train1,q)
        
        Y_pred=[int(gmm0.score_samples(X_test)[i]<gmm1.score_samples(X_test)[i]) for i in range(len(X_test))]
        
        print([list(confusion_matrix(Y_test,Y_pred)[i]) for i in range(len(confusion_matrix(Y_test,Y_pred)))],'\t',accuracy_score(Y_test,Y_pred))

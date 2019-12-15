from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

from sklearn import metrics
#digits=load_digits()
#data=scale(digits.data)
#true=digits.target
csv = pd.read_csv("Iris.csv")
species = csv["Species"]
data = csv.drop(columns = ["Species"],axis = 1)
pca = PCA(n_components=2).fit(data)
reduced_data = PCA(n_components=2).fit_transform(data)
wcss=[]
sco=[]
pursco = []
def kmean(d,k):
    global wcss
    global true
    print("___________________________________________________________________________")
    print("NUMBER OF CLUSTERS = ",k)
    print("___________________________________________________________________________")
    kmeans=KMeans(n_clusters=k, init='k-means++', max_iter=50, n_init=10, random_state=0)
    kmeans.fit(d)
    wcss.append(kmeans.inertia_)
    plt.scatter(d[:,0], d[:,1], c=kmeans.labels_.astype(float))
    plt.show()
    if(k==3):
        contingency_matrix = metrics.cluster.contingency_matrix(species, kmeans.labels_)
        print("PURITY SCORE =",np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix))
def gauss(d,k):
    global pursco
    print("___________________________________________________________________________")
    print("NUMBER OF GAUSSIAN COMPONENTS = ",k)
    print("___________________________________________________________________________")
    gmm=GaussianMixture(n_components=k)
    global sco
    gmm.fit(d)
    sco.append(gmm.score(d))
    lab=gmm.predict(d)
    plt.scatter(d[:,0], d[:,1], c=lab.astype(float))
    plt.show()
    contingency_matrix = metrics.cluster.contingency_matrix(species,lab)
    pursco.append(np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix))
    if k==3:
        print(contingency_matrix)
for i in [2,3,4,5,6,7,8,9]:
    kmean(reduced_data,i)
    gauss(reduced_data,i)
plt.title("ELBOW METHOD FOR K-MEANS CLUSTERING.")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.plot([2,3,4,5,6,7,8,9],wcss)
plt.show()
plt.title("ELBOW METHOD FOR GMM CLUSTERING.")
plt.xlabel('Number of clusters')
plt.ylabel('LOG-LIKELIHOOD.')
plt.plot([2,3,4,5,6,7,8,9],sco)
plt.show()
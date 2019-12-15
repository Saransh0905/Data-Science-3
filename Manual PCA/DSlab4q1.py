#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:21:20 2019

@author: saransh
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

class PCA(object):
    def __init__(self,size):
        self.size = size
        
        
    def generate_data(self,mean,covariance_matrix):
        self.x,self.y = np.random.multivariate_normal(mean,covariance_matrix,1000).T
        plt.scatter(self.y,self.x)
        plt.axis('equal')
        
    
    def find_eigen(self,covariance_matrix):
        self.data = np.array([self.x,self.y]).T
        print('size of data is:',sys.getsizeof(self.data))
        self.correlation_matrix = covariance_matrix
        eigen_pairs = np.linalg.eig(self.correlation_matrix)
        #print(self.correlation_matrix)
        #print(eigen_pairs)
        if eigen_pairs[0][0]>=eigen_pairs[0][1]:
            self.eigen_vector = eigen_pairs[1]
            
        else:
            self.eigen_vector = [eigen_pairs[1][1],eigen_pairs[1][0]]
        print('Eigen Vector:',self.eigen_vector)
        plt.quiver(0,0,self.eigen_vector[0],self.eigen_vector[1],scale = 2)
        #plt.show()
        
    def error_calculate(self,reduceDimension):
        self.error = 0
        
        for i in range(1000):
            
            An = np.matmul(self.eigen_vector,self.data[i].T)  
            #print(An)
            new_tuple = [0,0]
            for j in range(reduceDimension):
                new_tuple+= An[j]*self.eigen_vector[j] 
            error_tuple = 0
            for j in range(2):
                error_tuple += (self.data[i][j] - new_tuple[j])**2
            self.error+=error_tuple
        self.error = self.error/self.size
        print('value of error:',self.error)
            
            
    def return_components(self):
        data1x = []
        data1y = []
        data2x = []
        data2y = []
        for i in range(1000):
            
            An = np.matmul(self.eigen_vector,self.data[i].T)  
            #print(An)
            new_tuple1 = [0,0]
            new_tuple2 = [0,0]
            
            new_tuple1+= An[0]*self.eigen_vector[0] 
            data1x.append(new_tuple1[0])
            data1y.append(new_tuple1[1])

            new_tuple2+= An[1]*self.eigen_vector[1]
            data1x.append(new_tuple2[0])
            data1y.append(new_tuple2[1])            
        plt.scatter(data1x,data1y,color= 'red')
        plt.scatter(data2x,data2y,color = 'red')
        plt.show()                
                
                
            
            
            
        
mean1 = int(input())
mean2 = int(input()) 
varX = int(input())
varY = int(input())
covXY = int(input())
mean = [mean1,mean2]
co = [[varX,covXY],[covXY,varY]]
obj = PCA(2)
obj.generate_data(mean,co)
obj.find_eigen(co)
obj.error_calculate(1)
obj.error_calculate(2)
obj.return_components()


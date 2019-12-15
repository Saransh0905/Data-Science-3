import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import preprocessing

def read_data(path_to_file):
    return pd.read_csv(path_to_file)
	

def show_box_plot(attribute_name,dataframe):

    plt.boxplot(dataframe[attribute_name])
    plt.show()
    
    pass

def replace_outliers(dataframe):
	
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    med = dataframe.median()
    dt = dataframe.where((dataframe>lower) & (dataframe<upper),med,axis= 1)
    return dt
    pass

def count_outliers(dataframe):
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    med = dataframe.median()
    data = dataframe.where((dataframe<lower) & (dataframe<upper),axis= 1).count()
    return data

def range(dataframe,attribute_name):
    print("Maximum of ",attribute_name,max(dataframe[attribute_name]))
    print("Minimum of ",attribute_name,min(dataframe[attribute_name]),"\n")
    pass


def min_max_normalization(dataframe,range=None):
	
    dataframe = (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
    
    return dataframe
    
    pass
'''def standardize(dataframe):
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(dataframe)
    scaled_df = pd.DataFrame(scaled_df)
    return scaled_df
    pass
'''
def main():
    
    path_to_file="winequality_red_original.csv"
    dataframe=read_data(path_to_file)
    
    for col in dataframe.columns:
        if col!='quality':
            show_box_plot(col,dataframe)
    
    print(count_outliers(dataframe))
   
    print(replace_outliers(dataframe))
    for col in dataframe.columns:
        if col!='quality':
            show_box_plot(col,dataframe)
    
    for col in dataframe.columns:
        if col!='quality':
            range(dataframe,col)
    print(min_max_normalization(dataframe))
    print('gyguyggi',count_outliers(dataframe))
    return 
    

if __name__=="__main__":
	main()
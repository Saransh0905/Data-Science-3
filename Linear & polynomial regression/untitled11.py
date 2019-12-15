import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures 
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv("atmosphere_data.csv")

# comparison btw feature and label using scatter plot on original data

#Linear_regression 
feature = 'pressure'
label = 'temperature'
dataset.plot(x='pressure', y='temperature', style='o')  
plt.title("pressure vs temperature")  
plt.xlabel("pressure")  
plt.ylabel("temperature")  
plt.show()


X = dataset['pressure'].values

y = dataset['temperature'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("_____________")
print("LINEAR REGRESSION")
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print("intercept of the line",regressor.intercept_)
print("coef. of line",regressor.coef_)
y_pred = regressor.predict(X_test)
x_pred = regressor.predict(X_train)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

plt.scatter(X_test, y_test,  color='black')
plt.xlabel('pressure')
plt.ylabel('temperature')
plt.title('Linear Regression')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Root Mean Squared Error(for test data):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# if rmse for train data becomes 0, means over-fit.
print('Root Mean Squared Error(for training data)', np.sqrt(metrics.mean_squared_error(y_train, x_pred)))

plt.title(" scatter plot of actual temp. vs predicted temp.")
plt.xlabel("actual temp.")
plt.ylabel("predicted temp.")
plt.scatter(y_test,y_pred)
plt.show()






print("_________________")
print("nonlinear regression model using polynomial curve fitting")

X = dataset['pressure'].values.reshape(-1,1)

y = dataset['temperature'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rmse_te=[]
rmse_tr=[]
x_axis = np.arange(2,6)
for i in range(2,6):
    model = PolynomialFeatures(degree = i)
    X_ = model.fit_transform(X_train)
    X_ = pd.DataFrame(X_)
    
    X_test_ = model.fit_transform(X_test)
    X_test_ = pd.DataFrame(X_test_)
   
    regressor = LinearRegression() 
    regressor.fit(X_, y_train)
    y_pred = regressor.predict(X_test_)
    x_pred = regressor.predict(X_)
    rmse_test=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmse_te.append(rmse_test)
    rmse_train=np.sqrt(metrics.mean_squared_error(y_train, x_pred))
    rmse_tr.append(rmse_train)
    print("for degree",i)
    print('Root Mean Squared Error (for test data):',rmse_test)
    print('Root Mean Squared Error (for training data):',rmse_train)
    
    plt.scatter(X_train, y_train,  color='black')
    
    plt.xlabel('pressure')
    plt.ylabel('temperature')
    plt.scatter(X_train, x_pred, color='red', linewidth=2)
    
    plt.show()
    plt.title(" scatter plot of actual quality vs predicted quality")
    plt.xlabel("actual temp")
    plt.ylabel("predicted temp")
    plt.scatter(y_test,y_pred)
    plt.axis('equal')
    plt.show()
print("the bar graph of RMSE-train (y-axis) vs different values of degree of polynomial (x-axis).")
plt.bar(x_axis,rmse_tr)
plt.xlabel('Degree', fontsize=15)
plt.ylabel('Rmse value', fontsize=15)
plt.xticks(x_axis, fontsize=15, rotation=30)
plt.show()
    
print("the bar graph of RMSE-test (y-axis) vs different values of degree of polynomial (x-axis).")
plt.bar(x_axis,rmse_te)
plt.xlabel('Degree', fontsize=15)
plt.ylabel('Rmse value', fontsize=15)
plt.xticks(x_axis, fontsize=15, rotation=10)
plt.show()

    
print("___________")
print("Pearson correlation coeff_")
X = dataset[['humidity', 'pressure', 'rain', 'lightAvg', 'lightMax', 'moisture']]
y = dataset['temperature']

pearson = []
for i in range(len(X.columns)):
    pearson.append([st.pearsonr(X.iloc[:,i],y)[0],X.columns[i]])
pearson.sort(reverse=True)
X=dataset[[pearson[0][1],pearson[5][1]]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("_____________")
print("Multivariate linear regression after finding two most correlated attributes")
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
inter = regressor.intercept_ 
#print("coeff of all the features")
#print(coeff_df)

y_pred = regressor.predict(X_test)
x_pred = regressor.predict(X_train)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print('Root Mean Squared Error (for test data):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Root Mean Squared Error (for training data):', np.sqrt(metrics.mean_squared_error(y_train, x_pred)))

cq=list(coeff_df['Coefficient'])
c1 = cq[0]
c2 = cq[1]

def f(x,y):
    return np.array(c1*x+c2*y+inter)

x_axis = np.linspace(30,100,945)
y_ax = np.linspace(-1,101,945)



print("_____________")
print("Multivariate polynomial regression after finding two most correslated attributes")
for i in range(2,6):
    model = PolynomialFeatures(degree = i)
    X_ = model.fit_transform(X_train)
    X_ = pd.DataFrame(X_)
    
    X_test_ = model.fit_transform(X_test)
    X_test_ = pd.DataFrame(X_test_)
   
    regressor = LinearRegression() 
    regressor.fit(X_, y_train)
    y_pred = regressor.predict(X_test_)
    x_pred = regressor.predict(X_)
    rmse_test=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmse_train=np.sqrt(metrics.mean_squared_error(y_train, x_pred))
    print("for degree",i)
    print('Root Mean Squared Error (for test data):',rmse_test)
    print('Root Mean Squared Error (for training data):',rmse_train)
    
    
     
    plt.title(" scatter plot of actual temp vs predicted temp")
    plt.xlabel("actual temp")
    plt.ylabel("predicted temp")
    plt.scatter(y_test,y_pred)
    plt.axis('equal')
    plt.show()
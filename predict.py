# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *

# Importing the DS
dataset = pd.read_csv('corona.csv')
x = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values
y = y.reshape(len(y), 1)

print(dataset.head())

print(dataset.describe())
# Feature Scaling
from sklearn import preprocessing
sc_x = preprocessing.StandardScaler()
sc_y = preprocessing.StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole DS
from sklearn.svm import SVR
regressor = SVR(kernel='poly')
regressor.fit(x, y)

# Predicting a new result for 21.05.2020
a = str(sc_y.inverse_transform(regressor.predict(sc_x.transform([[97]]))))


# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the Polynomial Regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Covid-19 prediction')
plt.xlabel('Day')
plt.ylabel('Corona virus cases')
plt.show()
print("The total covid-19 cases predicted as on 27.05.2020="+a)



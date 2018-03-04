#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data_set = pd.read_csv('data.csv')
X = data_set.iloc[:, 1:2].values
Y = data_set.iloc[:, 2].values

#splitting the datset
""" from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) """

#Feature Scaling
""" from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """

#Fitting Linear Regression to Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#Fitting Polynomial Regression to Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
pol_lin_reg = LinearRegression()
pol_lin_reg.fit(x_poly, Y)

#Visualising Linear Regression
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Salary Vs Level (Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
len_X_grid = len(X_grid)
X_grid = X_grid.reshape(len_X_grid, 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, pol_lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Salary Vs Level (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predict new result with Linear Regression
print(lin_reg.predict(6.5))

#Predict new result with Polynmial Regression
print(pol_lin_reg.predict(poly_reg.fit_transform(6.5)))

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data_set = pd.read_csv('data.csv')
X = data_set.iloc[:, 1:2].values
Y = data_set.iloc[:, 2].values

#splitting the datset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Feature Scaling
""" from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """

#Fitting Regression to Dataset
#Create your regressor

#Predict new result with Regression
y_pred = regressor.predict(6.5)

#Visualising Polynomial Regression
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth Vs Bluff (Regression Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial Regression (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
len_X_grid = len(X_grid)
X_grid = X_grid.reshape(len_X_grid, 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth Vs Bluff (Regression Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
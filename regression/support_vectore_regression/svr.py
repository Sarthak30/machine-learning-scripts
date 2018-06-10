#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data_set = pd.read_csv('Position_Salaries.csv')
X = data_set.iloc[:, 1:2].values
Y = data_set.iloc[:, 2].values
Y = Y.reshape(-1, 1)

#splitting the datset
""" from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) """

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

#Fitting Support Vector Regression to Dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

#Predict new result with Regression
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising SVR Regression\
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth Vs Bluff (Regression Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
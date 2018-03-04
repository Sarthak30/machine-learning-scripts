#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data_set = pd.read_csv('50_Startups.csv')
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, 4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
one_hot_encoder_X = OneHotEncoder(categorical_features = [3])
X = one_hot_encoder_X.fit_transform(X).toarray()

#Avoid dummy variable trap
X = X[:, 1:]

#splitting the datset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#Feature Scaling
""" from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """

#Fitting multiple linear regression on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Prediciting the test set results
y_pred = regressor.predict(X_test)

#Building a optimal model using Backward elimination
import statsmodels.formula.api as sm
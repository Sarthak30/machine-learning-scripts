#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data_set = pd.read_csv('data.csv')
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, 3].values


#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer.fit(X[:, 1:3])  #upper bound is excluded
X[:, 1:3] = imputer.transform(X[:, 1:3])


#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
one_hot_encoder_X = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder_X.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


#splitting the datset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
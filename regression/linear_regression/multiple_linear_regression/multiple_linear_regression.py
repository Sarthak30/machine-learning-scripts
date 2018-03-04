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
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

def backward_elimination_with_p(y, x, sl):
    num_vars = len(x[0])
    for i in range(num_vars):
        regressor_ols = sm.OLS(endog=y, exog=x).fit()
        max_var =  max(regressor_ols.pvalues).astype(float)
        if max_var > sl:
            for j in range(0, num_vars-i):
                if (regressor_ols.pvalues[j].astype(float) == max_var):
                    x = np.delete(x, j, 1)
    regressor_ols.summary()
    return x

def backward_elimination_with_p_and_adjR2(y, x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
    
X_opt_with_only_p = backward_elimination_with_p(Y, X, 0.05)
X_opt = backward_elimination_with_p_and_adjR2(Y, X, 0.05)



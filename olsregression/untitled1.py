#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:02:36 2022

@author: christianfang
"""


import numpy as np
import pandas as pd
from scipy.stats import f

class OLSregression(object):

    def __init__(self):
        """This part initializes an empty array to store the coefficients"""
        self.coefficients = []
        
    def _reshape_x(self, X):
        """This function will be used in case there is only one independent 
        variable. The OLS calculation requires a 2-dimensional matrix 
        as input, so this function reshapes a one-dimensional feature to 
        a two-dimensional matrix"""
        return X.reshape(-1, 1)
    
    def _concatenate_ones(self, X):
        """This function creates a vector of ones with the same number of 
        elements as one column of the feature matrix has"""
        ones=np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)
    
    def fit(self, X, y):
        
        """The fit function estimates the OLS coefficients based on a 
        feature matrix (X) and an outcome variable (Y) as inputs
        """
        #if the feature matrix X is composed of only one column, we reshape it 
        if len(X.shape)== 1: X = self._reshape_x(X)
        #Appending a column of ones to the feature matrix
        X = self._concatenate_ones(X)
        #The following function applies the OLS method and stores the coefficients
        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        #

        #extract the intercept term from the coefficient list
        b0 = self.coefficients[0]
        #extract all other coefficients (the betas)
        other_betas = self.coefficients[1:]
        #Set the prediction to the intercept
        prediction = b0
        #Loop over all betas and values at the same time and increment the
        #prediction by the product of the two
        for xi, bi in zip(entry, other_betas): prediction += (bi * xi)
        return prediction
        
    def standarderrors(self, X, y, pred, coefficients):
        """this function calculates standard errors"""
        #Calculate parameters p and n
        p=len(coefficients)
        n=len(X)
        mse=(np.sum(np.square(Y-pred)))/(n-p)
        #Get matrix X*X
        X_with_intercept = np.empty(shape=(n, p), dtype=float)
        X_with_intercept[:, 0] = 1
        X_with_intercept[:, 1:p] = pd.DataFrame(X).values
        var_beta_hat = np.linalg.inv(X.T @ X) * mse 
        for p_ in range(len(coefficients)):
            ses=np.diag(var_beta_hat) ** 0.5
            return ses
    
    def rsquared(self, Y, pred):
        """This function calculates the R squared (Coefficient of Determination)
        of the fitted model"""
        #Caculate the residual sum of squares: the difference between observed
        #and predicted values, squared
        rss=np.sum(np.square((Y-pred)))
        #Calculate the mean of Y
        mean=np.mean(Y)
        #Calculate the sum of squares total: sum of the squared differences between Y and the mean of Y
        sst = np.sum(np.square(Y-mean))
        #Calculate the r_squared: 1 minus rss/sst
        r_squared = 1 - (rss/sst)
        return r_squared
    
        return r_squared
    def F(self, Y, pred, coefficients):
        #calculate p
        p=len(coefficients)
        #calculate n
        n=len(pred)
        #calculate mean
        mean=np.mean(Y)
        msm=(np.sum(np.square(pred-mean)))/(p-1)
        mse=(np.sum(np.square(Y-pred)))/(n-p)
        fval=msm/mse
        #Compute p-value of F using f.sf from scipy
        p_value = f.sf(fval, (p-1), (n-p))
        return fval, p_value
    def modelinfo(self, Y, pred, coefficients):
        #Calculate R squared
        #Caculate the residual sum of squares: the difference between observed
        #and predicted values, squared
        rss=np.sum(np.square((Y-pred)))
        #Calculate the mean of Y
        mean=np.mean(Y)
        #Calculate the sum of squares total: sum of the squared differences between Y and the mean of Y
        sst = np.sum(np.square(Y-mean))
        #Calculate the r_squared: 1 minus rss/sst
        r_squared = 1 - (rss/sst)
        #Calculate F
        #calculate p
        p=len(coefficients)
        #calculate n
        n=len(pred)
        #calculate mean
        msm=(np.sum(np.square(pred-mean)))/(p-1)
        mse=(np.sum(np.square(Y-pred)))/(n-p)
        fval=msm/mse
        #Compute p-value of F using f.sf from scipy
        p_value = f.sf(fval, (p-1), (n-p))
        return print("R squared: ", round(r_squared, 3), "F:", round(fval, 3), "p-value of F: ", p_value)
    

boston=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
boston.head()

X= boston.drop('medv', axis=1).values
Y= boston["medv"].values

#Workflow: Initialize the OLSregression model
model=OLSregression()
#Fit the model
model.fit(X, Y)
#Look at the coefficients! So pretty!
coefficients=model.coefficients

#significance
ses=model.standarderrors(X, Y, y_preds, coefficients)
ses
#Obtain and store the predicted values
y_preds = []
for row in X: y_preds.append(model.predict(row))
y_preds
#Calculate the r squared
model.rsquared(Y, y_preds)

#Calculate the f
model.F(Y, y_preds, coefficients)
model.modelinfo(Y, y_preds, coefficients)








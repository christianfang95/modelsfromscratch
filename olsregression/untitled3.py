#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:19:15 2022

@author: christianfang
"""

import numpy as np
import pandas as pd
from scipy.stats import f


class reg(object):

    def __init__(self, X, Y):
        """This part initializes an empty array to store the coefficients
        and calculates values used by all other methods"""
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
        
        """The fit function estimates the OLS coefficients, standard errors, 
        confidence intervals, p-values and model fit statistics based on a 
        feature matrix (X) and an outcome variable (Y) as inputs
        """
        #if the feature matrix X is composed of only one column, we reshape it 
        if len(X.shape)== 1: X = self._reshape_x(X)
        #Appending a column of ones to the feature matrix
        X = self._concatenate_ones(X)
        #The following function applies the OLS method and stores the coefficients
        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        #Calculating predicted values...
        def predict(self, entry):
            """This function generates predictions from the fitted models"""
            #extract the intercept term from the coefficient list
            b0 = self.coefficients[0]
            #extract all other coefficients (the betas)
            other_betas = self.coefficients[1:]
            #Set the prediction to the intercept
            self.prediction = b0
            #Loop over all betas and values at the same time and increment the
            #prediction by the product of the two
            for xi, bi in zip(entry, other_betas): self.prediction += (bi * xi)
        
        #standard errors
        #Calculate parameters p and n
        p=len(self.coefficients)
        n=len(X)
        mse=(np.sum(np.square(y-self.prediction)))/(n-p)
        #Get matrix X*X
        X_with_intercept = np.empty(shape=(n, p), dtype=float)
        X_with_intercept[:, 0] = 1
        X_with_intercept[:, 1:p+1] = pd.DataFrame(X).values
        var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * mse 
        for p_ in range(p):
            self.standarderrors=np.diag(var_beta_hat) ** 0.5
            
            
        #calculate 95% CIs
        self.ciupper=np.empty(shape=(p,1))
        for i in range(p):
           self.ciupper[i]=self.coefficients[i]+1.96*self.standarderrors[i]
        #lower
        self.cilower=np.empty(shape=(p,1))
        for i in range(p):
            self.cilower[i]=self.coefficients[i]-1.96*self.standarderrors[i]
            
            
        #Calculate R squared
        #Caculate the residual sum of squares: the difference between observed
        #and predicted values, squared
        rss=np.sum(np.square((y-self.prediction)))
        #Calculate the mean of Y
        mean=np.mean(y)
        #Calculate the sum of squares total: sum of the squared differences between Y and the mean of Y
        sst = np.sum(np.square(y-mean))
        #Calculate the r_squared: 1 minus rss/sst
        self.r_squared = 1 - (rss/sst)
        
        
        #Calculate F & P-value of F
        msm=(np.sum(np.square(self.prediction-mean)))/(p-1)
        mse=(np.sum(np.square(y-self.prediction)))/(n-p)
        self.fval=msm/mse
        #Compute p-value of F using f.sf from scipy
        self.p_value = f.sf(self.fvalfval, (p-1), (n-p))
    
#Workflow: Initialize the OLSregression model
model=reg(X, Y)
#Fit the model
model.fit(X, Y)


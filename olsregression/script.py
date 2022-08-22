#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:14:06 2022
@title: Multiple linear regression from scratch in Python
@author: christianfang
"""

#Import required packages
import pandas as pd
import numpy as np
from scipy.stats import f, t

#Load data and print head
boston=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
boston.head()

#Separate dependent variable (Y) from independent variables (X)
X= boston.drop('medv', axis=1).values
Y= boston["medv"].values

#Create list of column names
colnames = list(boston)
colnames.insert(0, 'Intercept')
del colnames[-1]
colnames

#Create dataframe that will hold the model
model = pd.DataFrame(index=colnames)
model.head()

#Calculating parameters N and p
N = len(X)
p = len(boston.columns)

#Constructing matrix X with intercept column
X_with_intercept = np.empty(shape=(N, p), dtype=float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = pd.DataFrame(X).values

#Estimate betas and assign them to column "B" of model
beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ pd.DataFrame(Y).values
model['B']=beta_hat

#Separate intercept estimate from coefficient estimates
intercept = beta_hat[0]
other_betas = beta_hat[1:]

#Calculate predicted values
y_hat = intercept + np.dot(X, other_betas)

#Calculate MSE
Ys=pd.DataFrame()
Ys["Yactual"]=Y
Ys["Ypred"]=y_hat
Ys.head()
mse=(np.sum(np.square(Ys["Yactual"]-Ys["Ypred"])))/(N-p)

#Calculate standard errors and assign them to column "SE"
var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * mse 
for p_ in range(len(model)):
    standarderrors = np.diag(var_beta_hat)**0.5
model["SE"]=standarderrors

#Calculate t values and assign them to column "t"
model['t']=model["B"]/model["SE"]

#Calculate p values and assign them to column "p"
p_values=np.around((t.sf(np.abs(model['t']), N-1)*2), 3)
model['p']=p_values

#Calculate 95% Confidence Intervlas and assign them to columns 
model['ci_lower']=model["B"]-1.96*model["SE"]
model['ci_upper']=model["B"]+1.96*model["SE"]

#Calculate model fit statistics: initialize dataframe that will hold statistics
modelfit=pd.DataFrame()

#Calculate R squared
rss=np.sum(np.square((Ys["Yactual"]-Ys["Ypred"])))
mean=np.mean(Ys["Yactual"])
sst = np.sum(np.square(Ys["Yactual"]-mean))
r_squared = 1 - (rss/sst)

# Calculate Adjusted r squared
r_sq_adj = 1- ((1-r_squared)*((N-1)/(N-p-1)))

#Calculate Root MSE
rmse=mse**0.5

#Calculate F and p-values of F
msm=(np.sum(np.square(Ys["Ypred"]-mean)))/(p-1)
fval=msm/mse
p_off = np.around(f.sf(fval, (p-1), (N-p)), 3)

#Add model fit values to dataframe
modelfitvals=[N, r_squared, r_sq_adj, rmse, fval, p_off]
colnames_modelfit=["Number of observations", "R sq", "Adjusted R sq", "Root MSE", "F", "Prob>F"]
modelinfo = pd.DataFrame(modelfitvals, index=colnames_modelfit)

#Print model  fit
modelinfo

#Print model
model
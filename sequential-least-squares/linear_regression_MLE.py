import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.formula.api as smf


np.random.seed(42)

size = 2500


def sim_data(sample_size = size, 
             beta_0 = 0.5, 
             beta_1 = -3, 
             beta_2 = 2):
             x1 = np.random.randn(sample_size)
             x2 = np.random.randn(sample_size)
             y = beta_0 + beta_1 * x1 + beta_2 * x2 + np.random.randn(sample_size)
             d = {'y' : y, 'x1' : x1, 'x2': x2}
             return pd.DataFrame(d)

dat = sim_data()

colnames = list(dat)
colnames.insert(0, 'Intercept')
del colnames[1]


# Set up model
model = pd.DataFrame(index = colnames)


N = len(dat)
#p = len(dat.columns)

x = pd.DataFrame(dat[['x1', 'x2']])
y = pd.DataFrame(dat['y'])
y = np.array(y)


#Design matrix add column of 1's to independent variables
X_design = np.hstack([np.ones((x.shape[0], 1)), x])

max_iter=100

tol=1e-5

#Initialize B 
B = np.array(np.zeros((X_design.shape[1],1)))

#loop through the Newton-Raphson algorithm
for i in range(max_iter):
    #Compute log likelihood
    y_hat = np.dot(X_design, B)
    ll = - np.sum(np.square(y - y_hat))    
    print('Iteration ' + str(i) + ' Log likelihood: ' + str(ll))
    #built diagonal matrix
    diag = np.diag(np.multiply(y_hat,(1-y_hat)).flatten())
    #parameter update rule
    Minv      = np.linalg.inv(np.matmul(np.matmul(X_design.T,diag),X_design))
    dB        = np.matmul(np.matmul(Minv,X_design.T),np.subtract(y,y_hat))
    B         += dB
    #check if we've reached the tolerance 
    if tol > np.linalg.norm(dB):
        break

y_hat = np.dot(X_design, B)


def grad_desc(Xs, Ys, rate = 0.0001, iterations = 1000):
    B = np.zeros((X_design.shape[1],1))
    for _ in range(iterations):
        error = Ys - np.dot(Xs, B)
        grad = -(Xs.T).dot(error)
        B = B - (grad)*rate
    return B



# The vector w representing all
# wieghts in the line
w = grad_desc(X_design, y)
w

import statsmodels.formula.api as smf

reg = smf.ols('y ~ x1 + x2', data = dat).fit()
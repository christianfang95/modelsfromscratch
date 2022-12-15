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
             eta = beta_0 + beta_1 * x1 + beta_2 * x2
             p = 1 / (1 + np.exp(-eta))
             y = np.random.binomial(1, p, size = sample_size)
             d = {'y' : y, 'x1' : x1, 'x2': x2}
             return pd.DataFrame(d)

dat = sim_data()

# Gat column names
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
    #compute conditional probabilities
    prob = 1/(1 + np.exp(-np.matmul(X_design, B)))
    prob = np.array(prob)
    #Log likelihood
    ll = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
    print('Iteration ' + str(i) + ' Log likelihood: ' + str(ll))
    #built diagonal matrix
    diag = np.diag(np.multiply(prob,(1-prob)).flatten())
    #parameter update rule
    Minv      = np.linalg.inv(np.matmul(np.matmul(X_design.T,diag),X_design))
    dB        = np.matmul(np.matmul(Minv,X_design.T),np.subtract(y,prob))
    B         += dB
    #check if we've reached the tolerance 
    if tol > np.linalg.norm(dB):
        break

#Calculate predicted probabilities
preds = np.array(1/(1 + np.exp(-np.matmul(X_design,B))))

# Weight matrix
W = np.diagflat(np.product(preds * (1-preds), axis=1))

# Covariance matrix
covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, W), X_design))

# Standard error
se = np.sqrt(np.diag(covLogit))

# Z scores
z = np.diag(np.array(B)/se)

#p-value
p_value = norm.sf(abs(z)) * 2


#FIll columns
model['B'] = np.around(np.array(B), 4)
model['SE'] = np.around(se, 3)
model['z'] = np.around(z, 3)
model['p'] = np.around(p_value, 3)
model['CI lower'] = np.around(model['B'] - 1.96*model['SE'], 3)
model['CI upper'] = np.around(model['B'] + 1.96*model['SE'], 3)
model



#LL null model
#Design matrix add column of 1's to independent variables
X_intercept = np.ones((x.shape[0], 1))

max_iter=100
tol=1e-5

#Initialize B 
B_int = np.array(np.zeros((X_intercept.shape[1],1)))

#loop through the Newton-Raphson algorithm
for i in range(max_iter):
    #compute conditional probabilities
    prob = 1/(1 + np.exp(-np.matmul(X_intercept, B_int)))
    prob = np.array(prob)
    #Log likelihood
    ll_null = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
    #built diagonal matrix
    diag = np.diag(np.multiply(prob,(1-prob)).flatten())
    #parameter update rule
    Minv = np.linalg.inv(np.matmul(np.matmul(X_intercept.T,diag),X_intercept))
    dB = np.matmul(np.matmul(Minv,X_intercept.T),np.subtract(y,prob))
    B_int  += dB
    #check if we've reached the tolerance 
    if tol > np.linalg.norm(dB):
        break

pseudo_r2 = 1-(ll/ll_null)

dict = {'Log-Likelihood': ll,
                             'Log-Likelihood Null': ll_null,
                             'Pseudo R-sq.': pseudo_r2}
model_summary = pd.DataFrame(dict, index = [1])



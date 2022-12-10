import numpy as np
import pandas as pd

x =pd.DataFrame(zip([0, 1.2, 3.2, -1],
                [1, 0, 1, 0]))
y = pd.DataFrame([0, 0, 1, 0])


def train(X,Y,max_iter=100,tol=1e-4):
        #add column of 1's to independent variables
        X_with_intercept = np.empty(shape = (N, p + 1), dtype = float)
        X_with_intercept[:, 0] = 1
        X_with_intercept[:, 1:p+1] = X.values
        #ensure correct dimensions for the input labels
        #Y       = Y.reshape(-1,1)
        #prepare parameters array
        B = np.zeros((X_with_intercept.shape[1],1))
        #loop through the Newton-Raphson algorithm
        for i in range(max_iter):
            #compute conditional probabilities
            P = 1/(1 + np.exp(-np.matmul(X_with_intercept,B)))
            #built diagonal matrix
            D = np.diag(np.multiply(P,(1-P)).flatten())
            #parameter update rule
            Minv      = np.linalg.inv(np.matmul(np.matmul(X_with_intercept.T,D),X_with_intercept))
            dB        = np.matmul(np.matmul(Minv,X_with_intercept.T),np.subtract(Y,P))
            B         += dB
            #check if we've reached the tolerance 
            if tol > np.linalg.norm(dB):
                break
        return(B)

b = train(x, y)

ones  = np.ones(x.shape[0])
x[:,0:] = ones

x_with_intercept = x[:, 0] = 1

N = len(x)
p = len(x.columns)
X_with_intercept = np.empty(shape = (N, p + 1), dtype = float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, p] = x.values
X_with_intercept


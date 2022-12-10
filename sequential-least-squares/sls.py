import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# Set random seed
np.random.seed(42)

# Define function to simulate data

def prob2logit(prob):
  logit = np.log(prob / (1 - prob))
  return logit


intercept = prob2logit(0.8)


def sim_data(sample_size = 1000, 
             beta_0 = intercept, 
             beta_1 = -3):
             x = np.random.randn(sample_size)
             eta = beta_0 + beta_1 * x 
             p = 1 / (1 + np.exp(-eta))
             y = np.random.binomial(1, p, size = sample_size)
             d = {'y' : y, 'x' : x}
             return pd.DataFrame(d)


# Define function for sequential least squares

def SLS(data = data):
    dat = data
    reg = smf.ols("y ~ x", data=dat).fit()
    fitted = reg.fittedvalues
    i = 1
    if (min(fitted) > 0) & (max(fitted) < 1):
        print('No out of range predictions. OLS is equivalent to SLS.')
    while min(fitted < 0) | max(fitted > 1):
        print('Iteration ' + str(i) + '; Remaining observations: ' +
              str(len(dat)) + '; Invalid Pr(Y=1): ' + str(round(100 * ((sum(fitted < 0) +
                                                                        sum(fitted > 1)) / len(dat)), 2)) + ' %')
        i += 1
        dat = dat[(fitted < 1) & (fitted > 0)]
        reg = smf.ols("y ~ x", data=dat).fit()
        fitted = reg.fittedvalues
    fitted = reg.fittedvalues
    dat["fitted"] = fitted
    return dat


def goldberger (data = data):
    dat = data
    reg = smf.ols("y ~ x", data=dat).fit()
    fitted = reg.fittedvalues
    fitted[fitted < 0] = 0.001
    fitted[fitted > 1] = 0.999
    weight = np.sqrt(1 / (fitted * (1 - fitted)))
    reg = smf.wls('y ~ x', data = dat, weights = weight).fit().params['x']
    data['WLS_fitted'] = smf.wls('y ~ x', data = dat, weights = weight).fit().fittedvalues
    WLS = pd.DataFrame()
    WLS['weight'] = weight
    return reg, weight


# Get coefficients

def get_coefficients(data = data):
    data = data
    res = SLS(data = data)
    gws, WLS = goldberger()
    seq = smf.ols("y ~ x", data=res).fit().params['x']
    lpm = smf.ols("y ~ x", data=data).fit().params['x']
    logit = smf.logit("y ~ x", data=data).fit() \
                                     .get_margeff(at='overall', method = 'dydx') \
                                     .summary_frame()['dy/dx']['x']
    data['lpm_fitted'] = smf.ols("y ~ x", data=data).fit().fittedvalues
    data['logit_fitted'] = smf.logit("y ~ x", data=data).fit().predict()
    return seq, lpm, logit, gws, WLS, res, data


# Graph results

def plot(data = data, res = res, lpm = lpm, logit = logit, seq = seq, gws = gws):
    fig, ax1 = plt.subplots(nrows = 1, ncols= 1)
    l1 = sns.lineplot(ax = ax1, x = data['x'], y = data['lpm_fitted'], label = 'LPM, AME: ' + str(round(lpm, 2)))
    l2 = sns.lineplot(ax = ax1, x = data['x'], y = data['logit_fitted'], label = 'Logistic, AME: ' + str(round(logit, 2)))
    l3 = sns.lineplot(ax = ax1, x = res['x'], y = res['fitted'], label = 'SLS, AME: ' + str(round(seq, 2)))
    l3 = sns.lineplot(ax = ax1, x = res['x'], y = data['WLS_fitted'], label = 'WLS, AME: ' + str(round(gws, 2)))
    ax1.set_title('Predicted values vs. x')
    ax1.set_ylabel('Predicted value')
    plt.show()

# Main function

def main():
    data = sim_data()
    seq, lpm, logit, gws, WLS, res, data = get_coefficients()
    return data, seq, lpm, logit, gws, WLS, res

data, seq, lpm, logit, gws, WLS, res = main()

plot(data = data, res = res, lpm = lpm, logit = logit, seq = seq, gws = gws)


data = sim_data()

seq = 
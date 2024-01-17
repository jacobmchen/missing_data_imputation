"""
Code implementing the data-generating process specified in Josse et al.

Created: January 15, 2024

Last updated: January 16, 2024
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import pickle

def generate_data(size=1000, rho=0.5, d=10, num_missing=3, p=0.8, imputation_value=[0]*10, DGP='quadratic', missing='MCAR'):
    """
    This function controls the data-generating process.

    imputation_value is a list to indicate the value of the inputted value for each X

    num_missing indicates how many variables in the vector X will suffer from missing values;
    the first num_missing variables will suffer from missingness

    note: when imputation value is mean, this function also returns the mean of X1
    to use in the imputation of the test distribution
    """
    # data generating process for models 1-3

    # d-dimensional vector of 1s
    mu = [1]*d
    
    # add together two matrices: first one is a dxd matrix composed of just rho
    # second one is a dxd identity matrix multiplied by (1-rho)
    sigma = np.add(np.asmatrix([[rho]*d]*d), (1-rho) * np.identity(d))
    
    X_vec = np.random.multivariate_normal(mu, sigma, size)

    # separate the generated vector into individual data columns by storing the values 
    # into a list of lists
    X = []
    for i in range(d):
        X.append(np.array(X_vec[:, i]))

    # create a list where R[i] is the missingness indicator for X[i]
    R = []

    # induce missingness
    # value of 1 in R[i] indicates observed
    if missing == 'MNAR':
        # only keep values above 20th percentile as p = 0.8
        
        for i in range(num_missing):
            # create an array for the missingness indicator
            R_i = []
            # find the 1-pth quantile of the array
            q = np.quantile(X[i], 1-p)
            # only keep values that are above the 1-pth quantile of the array
            for i in range(len(X[i])):
                if X[0][i] > q:
                    R_i.append(1)
                else:
                    R_i.append(0)
            
            R.append(R_i)
    elif missing == 'MCAR':
        # there is a 20% chance of observing the value of X[i]
        # 20% of missing values is consistent with the paper

        # note: when the DGP is linear, Friedman, or nonlinear, the missingness mechanism
        # is always MCAR

        for i in range(num_missing):
            # append the entire array to R
            R.append(np.random.binomial(1, p, size))
    elif missing == 'predictive':
        # a pattern mixture model where the missingness indicator is a part
        # of the regression function
        # note that the predictive missingness mechanism is only applicable for model 1

        # first, the missing values are generated from a Bernoulli distribution like in the 
        # MCAR case
        for i in range(num_missing):
            # append the entire array to R
            R.append(np.random.binomial(1, p, size))
    else:
        print('invalid')
        return
    
    # decide how the outcome variable Y is calculated
    if DGP == 'quadratic' and missing != 'predictive':
        # quadratic DGP
        Y = X[0]**2 + np.random.normal(0, 0.1, size)
    elif DGP == 'quadratic' and missing == 'predictive':
        # a pattern mixture model where the missingness indicator is a part
        # of the regression function

        # this missingness mechanism should only be possible when the DGP is set to quadratic
        Y = np.random.normal(0, 0.1, size)
        for i in range(num_missing):
            Y += X[i]**2 + 2*R[i]
    elif DGP == 'linear':
        # linear DGP
        Y = np.random.normal(0, 0.1, size)
        beta = [1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3]
        for i in range(d):
            Y += beta[i]*X[i]
    else:
        print('invalid')
        return

    # create the dataframe with all the variables
    data = pd.DataFrame({'Y': Y})
    for i in range(d):
        data['X'+str(i+1)] = X[i]
    for i in range(num_missing):
        data['R'+str(i+1)] = R[i]

    # impute the missing values
    if imputation_value == 'mean':
        # find the mean of the observed values for each variable
        mean = []

        for i in range(num_missing):
            data_copy = data.copy()
            data_copy = data_copy[data_copy['R' + str(i+1)] == 1]
            mean.append(np.mean(data_copy['X' + str(i+1)]))
        # print('mean:', mean)
    elif imputation_value == 'delete_rows':
        # only keep rows of data where every single value of X is observed
        data_copy = data.copy()

        for i in range(num_missing):
            data_copy = data_copy[data_copy['R' + str(i+1)] == 1]

        return data_copy
    elif imputation_value == 'nan':
        data_copy = data.copy()
        for i in range(num_missing):
            for j in range(len(data_copy['X' + str(i+1)])):
                if data_copy.at[j, 'R' + str(i+1)] == 0:
                    # data_copy['X' + str(i+1)][j] = np.nan
                    data_copy.at[j, 'X' + str(i+1)] = np.nan
        return data_copy
    elif imputation_value == 'no_missing':
        return data

    for j in range(num_missing):
        for i in range(len(R[j])):
            # replace missing values with the mean or the designated imputation value
            if R[j][i] == 0:
                if imputation_value == 'mean':
                    X[j][i] = mean[j]
                else:
                    X[j][i] = imputation_value[j]

    # print('proportion of observed X1:', np.mean(R1))
    
    # after imputting missing values, put the partially observed variables back into the
    # dataframe so that they are actually udpated with the imputed values
    data = pd.DataFrame({'Y': Y})
    for i in range(d):
        data['X'+str(i+1)] = X[i]
    for i in range(num_missing):
        data['R'+str(i+1)] = R[i]

    if imputation_value == 'mean':
        return data, mean
    else:
        return data
    

if __name__ == "__main__":
    np.random.seed(0)

    print(generate_data(missing="predictive"))
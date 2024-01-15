import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import pickle

def generate_data(size=1000, rho=0.5, d=10, num_missing=3, imputation_value=[0]*10, DGP='quadratic', missing='MCAR'):
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

    # decide how the outcome variable Y is calculated
    if DGP == 'quadratic':
        # quadratic DGP
        Y = X[0]**2 + np.random.normal(0, 0.1, size)
    elif DGP == 'linear':
        # linear DGP
        Y = np.random.normal(0, 0.1, size)
        beta = [1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3]
        for i in range(d):
            Y += beta[i]*X[i]

    # create a list where R[i] is the missingness indicator for X[i]
    R = []

    # induce missingness
    # value of 1 in R[i] indicates observed
    if missing == 'MNAR':
        print('invalid')
        return
        # in progress
        # only keep values above 20th percentile as p = 0.8
        # R1 = []
        # q20 = np.quantile(X[0], 0.2)
        # for i in range(len(X[0])):
        #     if X[0][i] > q20:
        #         R1.append(1)
        #     else:
        #         R1.append(0)
    elif missing == 'MCAR':
        # there is a 20% chance of observing the value of X[i]
        # 20% of missing values is consistent with the paper

        for i in range(num_missing):
            # append the entire array to R
            R.append(np.random.binomial(1, 0.8, size))
    elif missing == 'predictive':
        # a pattern mixture model where the missingness indicator is a part
        # of the regression function
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
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

from em_imputation import em_imputation

# Import necessary packages to use R in Python
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

def create_matrix(data, mask):
    # helper function for creating a data matrix in the format
    # required for xgboost

    vars = data.columns

    matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(vars)):
            if vars[j][0] == 'R':
                if mask:
                    row.append(data.at[i, vars[j]])
            elif vars[j][0] == 'Y':
                continue
            else:
                row.append(data.at[i, vars[j]])
        # if mask:
        #     for j in range(num_missing):
        #         row.append(data.at[i, 'R'+str(j+1)])
        matrix.append(row)

    return matrix

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
        # quadratic DGP, use the first three variables in the DGP
        Y = X[0]**2 + X[1]**2 + X[2]**2 + np.random.normal(0, 0.1, size)
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
                    data_copy.at[j, 'X' + str(i+1)] = np.nan
        return data_copy
    elif imputation_value == 'mia':
        data_copy = data.copy()

        # step 1: make two copies of the columns of data that are incomplete
        for i in range(num_missing):
            data_copy['X'+str(i+1)+'copy1'] = data_copy['X'+str(i+1)].copy()
            data_copy['X'+str(i+1)+'copy2'] = data_copy['X'+str(i+1)].copy()

        # step 2: in the first copy, impute the missing values with negative infinity
        for i in range(num_missing):
            for j in range(len(data_copy['X'+str(i+1)+'copy1'])):
                if data_copy.at[j, 'R'+str(i+1)] == 0:
                    data_copy.at[j, 'X'+str(i+1)+'copy1'] = -10**10

        # step 3: in the second copy, impute the missing values with positive infinity
        for i in range(num_missing):
            for j in range(len(data_copy['X'+str(i+1)+'copy2'])):
                if data_copy.at[j, 'R'+str(i+1)] == 0:
                    data_copy.at[j, 'X'+str(i+1)+'copy2'] = 10**10

        # step 4: delete the original column of data
        for i in range(num_missing):
            data_copy = data_copy.drop(columns=['X'+str(i+1)])

        # step 5: return the updated dataset
        return data_copy
    elif imputation_value == 'gaussian':
        print(data)
        # step 1: fill in nan values in the original dataset
        data_copy = data.copy()
        for i in range(num_missing):
            for j in range(len(data_copy['X' + str(i+1)])):
                if data_copy.at[j, 'R' + str(i+1)] == 0:
                    data_copy.at[j, 'X' + str(i+1)] = np.nan
        print(data_copy)

        # step 2: convert the dataset into the format necessary for the norm package
        data_copy = data_copy.drop(columns=['R1', 'R2', 'R3', 'Y'])
        vector = []
        for i in range(d):
            vals = np.array(data_copy['X'+str(i+1)])
            for j in range(len(vals)):
                vector.append(vals[j])
        vector = robjects.FloatVector(vector)
        matrix = robjects.r['matrix'](vector, nrow=size)

        # step 3: use the norm package to estimate mu and sigma
        # import the package
        norm_package = importr('norm')

        prelim = robjects.r['prelim.norm']
        s = prelim(matrix)

        em = robjects.r['em.norm']
        thetahat = em(s)

        getparam = robjects.r['getparam.norm']
        output = getparam(s, thetahat, corr=False)
        mu_hat = output[0]
        sigma_hat = output[1]

        data_copy = em_imputation(data, create_matrix(data_copy, False), mu_hat, sigma_hat, d, num_missing)

        return data_copy
    elif imputation_value == 'no_missing':
        return data

    # the following code only applies to the mean imputation method or specifying a
    # custom imputation value
    for j in range(num_missing):
        for i in range(len(R[j])):
            # replace missing values with the mean or the designated imputation value
            if R[j][i] == 0:
                if imputation_value == 'mean':
                    X[j][i] = mean[j]
                else:
                    X[j][i] = imputation_value[j]
    
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

    pandas2ri.activate()

    print(generate_data(missing="MCAR", imputation_value='gaussian'))
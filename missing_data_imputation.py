"""
Simulations to test if imputing by a constant is sufficient to train a supervised
classifier according to paper by Julie et al.

Created: November 9, 2023

Last updated: November 26, 2023

To-do:
1. double check mean imputation uses training set mean as the imputed value
2. implement other methods in the paper
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

def generate_data(size=1000, rho=0.5, d=10, imputation_value=[0]*10, DGP='quadratic', missing='MAR'):
    """
    This function controls the data-generating process.

    imputation_value is a list to indicate the value of the inputted value for each X

    note: when imputation value is mean, this function also returns the mean of X1
    to use in the imputation of the test distribution
    """
    # data generating process for models 1-3

    # d-dimensional vector of 1s
    mu = [1]*d
    
    # add together two matrices: first one is a dxd matrix composed of just rho
    # second one is a dxd identity matrix multiplied by (1-rho)
    sigma = np.add([[rho]*d]*d , (1-rho) * np.identity(d))
    
    X_vec = np.random.multivariate_normal(mu, sigma, size)

    # separate the generated vector into individual data columns by storing the values 
    # into a list of lists
    X = []
    for i in range(10):
        X.append(X_vec[:, i])

    # decide how the outcome variable Y is calculated
    if DGP == 'quadratic':
        # quadratic DGP
        Y = X[0]*X[0] + np.random.normal(0, 0.1, size)
    elif DGP == 'linear':
        # linear DGP
        Y = np.random.normal(0, 0.1, size)
        beta = [1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3]
        for i in range(10):
            Y += beta[i]*X[i]

    # create a list where R[i] is the missingness indicator for X[i]
    R = []

    # induce missingness
    # value of 1 in R[i] indicates observed
    if missing == 'MNAR':
        pass
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

        for i in range(d):
            # append a dummy value so that the array does not go out of bounds
            R.append(0)
            R[i] = np.random.binomial(1, 0.8, size)
    elif missing == 'predictive':
        # a pattern mixture model where the missingness indicator is a part
        # of the regression function
        pass

    data = pd.DataFrame({'X1': X[0], 'X2': X[1], 'X3': X[2], 'X4': X[3], 'X5': X[4], 'X6': X[5],
                         'X7': X[6], 'X8': X[7], 'X9': X[8], 'X10': X[9], 'Y': Y, 'R1': R[0],
                         'R2': R[1], 'R3': R[2], 'R4': R[3], 'R5': R[4], 'R6': R[5], 'R7': R[6],
                         'R8': R[7], 'R9': R[8], 'R10': R[9]})

    # impute the missing values
    if imputation_value == 'mean':
        # find the mean of the observed values for each variable
        mean = []

        for i in range(d):
            data_copy = data.copy()
            data_copy = data_copy[data_copy['R' + str(i+1)] == 1]
            mean.append(np.mean(data_copy['X' + str(i+1)]))
        # print('mean:', mean)
    elif imputation_value == 'delete_rows':
        # only keep rows of data where every single value of X is observed
        data_copy = data.copy()

        for i in range(d):
            data_copy = data_copy[data_copy['R' + str(i+1)] == 1]

        return data_copy
    elif imputation_value == 'nan':
        data_copy = data.copy()
        for i in range(d):
            for j in range(len(data_copy['X' + str(i+1)])):
                if data_copy['R' + str(i+1)][j] == 0:
                    data_copy['X' + str(i+1)][j] = np.nan
        return data_copy
    elif imputation_value == 'no_missing':
        return data

    for j in range(d):
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
    data = pd.DataFrame({'X1': X[0], 'X2': X[1], 'X3': X[2], 'X4': X[3], 'X5': X[4], 'X6': X[5],
                         'X7': X[6], 'X8': X[7], 'X9': X[8], 'X10': X[9], 'Y': Y, 'R1': R[0],
                         'R2': R[1], 'R3': R[2], 'R4': R[3], 'R5': R[4], 'R6': R[5], 'R7': R[6],
                         'R8': R[7], 'R9': R[8], 'R10': R[9]})

    if imputation_value == 'mean':
        return data, mean
    else:
        return data

def train_model(data, DGP='quadratic', model='regression', mask=False):
    """
    This function trains a predictor.
    """

    if DGP == 'quadratic':
        formula = 'Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10'
    elif DGP == 'linear':
        formula = 'Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10'

    # include the missingness indicator in the regression
    if mask:
        formula += '+R1'

    if model == 'regression':
        trained_model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
        return trained_model
    elif model == 'decision_tree':
        # use the sklearn decision tree

        # make sure the decision tree regressor has the same hyper parameters as the rpart function in R
        regr = DecisionTreeRegressor(max_depth=30, min_samples_split=20, min_samples_leaf=7, ccp_alpha=0.01)

        data_X = data.drop(columns=['Y'])
        regr.fit(data_X, data['Y'])
        return regr
    
def run_experiments(repetitions=1000, verbose=False):
    """
    This function runs the synthetic experiments using four scenarios:
    1. no missing values as a baseline
    2. dropping rows of missing values
    3. imputing by the mean of the missing variable in question
    4. imputing by some out of range value

    It repeats each scenario by the specified amount of times and reports
    the average of the R^2 values.
    """
    r2_values = [[], [], [], []]

    model_type = 'decision_tree'
    DGP = 'quadratic'
    missing_mechanism = 'MCAR'
    print(model_type, DGP, missing_mechanism, 'size='+str(size))
    print()

    for i in range(repetitions):
        """
        calculate baseline
        """
        data_train = generate_data(size=size, imputation_value='no_missing', DGP=DGP, missing=missing_mechanism)
        
        model = train_model(data_train, DGP=DGP, model=model_type, mask=False)
        # print(model.summary())

        data_test = generate_data(size=size, imputation_value='no_missing', DGP=DGP, missing=missing_mechanism)

        if model_type == 'regression':
            predictions = model.predict(data_test)
        elif model_type == 'decision_tree':
            data_X = data_test.drop(columns=['Y'])
            predictions = model.predict(data_X)
        
        r2_values[0].append(r2_score(data_test['Y'], predictions))
        if verbose:
            print('baseline no missingness: R^2 score', r2_score(data_test['Y'], predictions))
            print()

        """
        drop rows of missing values
        """
        data_train = generate_data(size=size, imputation_value='delete_rows', DGP=DGP, missing=missing_mechanism)
        
        model = train_model(data_train, DGP=DGP, model=model_type, mask=False)
        # print(model.summary())

        data_test = generate_data(size=size, imputation_value='delete_rows', DGP=DGP, missing=missing_mechanism)

        if model_type == 'regression':
            predictions = model.predict(data_test)
        elif model_type == 'decision_tree':
            data_X = data_test.drop(columns=['Y'])
            predictions = model.predict(data_X)
        
        r2_values[1].append(r2_score(data_test['Y'], predictions))
        if verbose:
            print('drop rows of missing data: R^2 score', r2_score(data_test['Y'], predictions))
            print()

        """
        mean imputation
        """
        data_train, mean = generate_data(size=size, imputation_value='mean', DGP=DGP, missing=missing_mechanism)
        
        model = train_model(data_train, DGP=DGP, model=model_type, mask=False)
        # print(model.summary())

        # impute the testing dataset by the mean of the training dataset
        data_test = generate_data(size=size, imputation_value=mean, DGP=DGP, missing=missing_mechanism)

        if model_type == 'regression':
            predictions = model.predict(data_test)
        elif model_type == 'decision_tree':
            data_X = data_test.drop(columns=['Y'])
            predictions = model.predict(data_X)

        r2_values[2].append(r2_score(data_test['Y'], predictions))
        if verbose:
            print('mean imputation: R^2 score', r2_score(data_test['Y'], predictions))
            print()

        """
        out of range imputation
        """
        data_train = generate_data(size=size, imputation_value=[99999]*10, DGP=DGP, missing=missing_mechanism)
        
        model = train_model(data_train, DGP=DGP, model=model_type, mask=False)
        # print(model.summary())

        data_test = generate_data(size=size, imputation_value=[99999]*10, DGP=DGP, missing=missing_mechanism)

        if model_type == 'regression':
            predictions = model.predict(data_test)
        elif model_type == 'decision_tree':
            data_X = data_test.drop(columns=['Y'])
            predictions = model.predict(data_X)

        r2_values[3].append(r2_score(data_test['Y'], predictions))
        if verbose:
            print('out of range imputation: R^2 score', r2_score(data_test['Y'], predictions))
            print()

    return (np.mean(r2_values[0]), np.mean(r2_values[1]), np.mean(r2_values[2]), np.mean(r2_values[3]))

if __name__ == "__main__":
    size = 1000

    np.random.seed(0)

    # print(run_experiments())

    # code below is for testing purposes
    DGP = 'quadratic'
    missing_mechanism = 'MCAR'
    model_type = 'decision_tree'
    """
    mean imputation
    """
    data_train, mean = generate_data(size=size, imputation_value='mean', DGP=DGP, missing=missing_mechanism)
    
    # model = train_model(data_train, DGP=DGP, model=model_type, mask=False)
    # # print(model.summary())

    # data_test = generate_data(size=size, imputation_value=mean, DGP=DGP, missing=missing_mechanism)

    # if model_type == 'regression':
    #     predictions = model.predict(data_test)
    # elif model_type == 'decision_tree':
    #     data_X = data_test.drop(columns=['Y'])
    #     predictions = model.predict(data_X)

    # print(r2_score(data_test['Y'], predictions))

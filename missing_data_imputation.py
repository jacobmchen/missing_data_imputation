"""
Simulations to test if imputing by a constant is sufficient to train a supervised
classifier according to paper by Julie et al.

Created: November 9, 2023

Last updated: January 9, 2024

To-do:
1. implement other methods in the paper
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import pickle

# Import necessary packages to use R in Python
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula

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

def run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, imputation_value):
    """
    This function runs a single experiment and calculates a single R^2 value depending on
    the imputation method and model_type.

    Possible imputation_values:
    int array representing the imputation values
    no_missing
    mean
    delete_rows
    nan
    """
    if imputation_value == 'mean':
        data_train, mean = generate_data(size=size, d=d, imputation_value=imputation_value, DGP=DGP, missing=missing_mechanism)
    else:
        data_train = generate_data(size=size, d=d, imputation_value=imputation_value, DGP=DGP, missing=missing_mechanism)
    
    model = train_model(data_train, d=d, DGP=DGP, model=model_type, mask=mask)
    # print(model.summary())

    if imputation_value == 'mean':
        # use the calculated mean values from the training dataset as the imputation values in the testing dataset
        data_test = generate_data(size=size, d=d, imputation_value=mean, DGP=DGP, missing=missing_mechanism)
    else:
        data_test = generate_data(size=size, d=d, imputation_value=imputation_value, DGP=DGP, missing=missing_mechanism)

    predictions = make_predictions(data_test, d, model_type, mask, model)
    # fill NaN values with the mean in case a model ever makes a NaN prediction
    # (this seems to only happen with the ctree model)
    # calculate the mean
    cnt = 0
    sum = 0
    for i in range(len(predictions)):
        if not np.isnan(predictions[i]):
            cnt += 1
            sum += predictions[i]
    mean = sum / cnt

    for i in range(len(predictions)):
        if np.isnan(predictions[i]):
            predictions[i] = mean
    # predictions = list(pd.Series(predictions).fillna(np.mean(predictions)))

    return r2_score(data_test['Y'], predictions)

def train_model(data, d=9, num_missing=3, DGP='quadratic', model='regression', mask=False):
    """
    This function trains a predictor.
    """

    if model == 'regression':
        formula = 'Y~'
        for i in range(d):
            formula += '+X' + str(i+1)

        # include the missingness indicator in the regression
        if mask:
            for i in range(num_missing):
                formula += '+R' + str(i+1)

        trained_model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
        return trained_model
    elif model == 'decision_tree':
        # use the sklearn decision tree

        # make sure the decision tree regressor has the same hyper parameters as the rpart function in R
        regr = DecisionTreeRegressor(max_depth=30, min_samples_split=20, min_samples_leaf=7, ccp_alpha=0.01)

        data_X = data.drop(columns=['Y'])
        if not mask:
            for i in range(num_missing):
                data_X = data_X.drop(columns=['R' + str(i+1)])
        regr.fit(data_X, data['Y'])
        return regr
    elif model == 'rpart':
        # import the rpart package from R which will allow us to train an rpart model
        rpart_package = importr('rpart')

        # create the formula
        formula = 'Y~'
        for i in range(d):
            formula += '+X' + str(i+1)

        # include the missingness indicator in the regression
        if mask:
            for i in range(num_missing):
                formula += '+R' + str(i+1)

        # train a decision tree, and use all default values as specified in Josse et al.
        tree = rpart_package.rpart(formula=Formula(formula), data=data)
        # print(tree)

        return tree
    elif model == 'ctree':
        # import the partykit package from R which contains the ctree implementation
        partykit_package = importr('partykit')

        # create the formula
        formula = 'Y~'
        for i in range(d):
            formula += '+X' + str(i+1)

        # include the missingness indicator in the regression
        if mask:
            for i in range(num_missing):
                formula += '+R' + str(i+1)
        
        # train a ctree using all default values
        tree = partykit_package.ctree(formula=Formula(formula), data=data)

        return tree

        
def make_predictions(data_test, d, model_type, mask, model):
    """
    Given a trained model and testing data, make predictions for the outcome
    variable.
    """

    if model_type == 'regression':
        predictions = model.predict(data_test)
    elif model_type == 'decision_tree':
        data_X = data_test.drop(columns=['Y'])
        if not mask:
            for i in range(d):
                data_X = data_X.drop(columns=['R' + str(i+1)])
        predictions = model.predict(data_X)
    elif model_type == 'rpart' or model_type == 'ctree':
        predictions = robjects.r.predict(model, newdata=data_test)
        predictions = list(predictions)

    return predictions
    
def run_experiments(repetitions=1000, verbose=False):
    """
    This function runs the synthetic experiments using four scenarios:
    1. no missing values as a baseline
    2. dropping rows of missing values
    3. imputing by the means of the missing variables
    4. imputing by some out of range value

    It repeats each scenario by the specified amount of times and reports
    the average of the R^2 values.
    """
    r2_values = [[], [], [], [], [], [], [], [], []]

    size = 2000
    np.random.seed(0)

    model_type = 'rpart'
    DGP = 'quadratic'
    missing_mechanism = 'MCAR'
    d = 9
    print(model_type, 'and ctree;', DGP, missing_mechanism, 'size='+str(size), 'repetitions=', repetitions)
    print()

    for i in range(repetitions):
        cnt = 0

        """
        calculate baseline
        """
        mask = False
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'no_missing'))
        cnt += 1

        """
        mean imputation
        """
        mask = False
        # during mean imputation, we obtain a vector of means for each of X_i from the training dataset to 
        # use as the imputation value in the testing dataset
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'mean'))
        cnt += 1

        """
        mean imputation with mask
        """
        mask = True
        # during mean imputation, we obtain a vector of means for each of X_i from the training dataset to 
        # use as the imputation value in the testing dataset
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'mean'))
        cnt += 1

        """
        out of range imputation
        """
        mask = False
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, [99999]*d))
        cnt += 1

        """
        out of range imputation with mask
        """
        mask = True
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, [99999]*d))
        cnt += 1

        """
        CART with surrogate splits with no mask
        """
        mask = False
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'nan'))
        cnt += 1

        """
        CART with surrogate splits with mask
        """
        mask = True
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'nan'))
        cnt += 1

        """
        ctree with no mask
        """
        mask = False
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, 'ctree', 'nan'))
        cnt += 1

        """
        ctree with mask
        """
        mask = True
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, 'ctree', 'nan'))
        cnt += 1

    return [np.mean(r2_values[i]) for i in range(9)], r2_values

if __name__ == "__main__":
    # Must be activated to use R packages in Python
    pandas2ri.activate()

    result = run_experiments(repetitions=1000)
    print('mean values', result[0])
    print('baseline, mean, mean with mask, out of range, out of range with mask, CART with surrogate splits',
          'CART with surrogate splits with mask', 'ctree no mask', 'ctree with mask')
    
    output = pd.DataFrame({'mean': result[1][1], 'mean + mask': result[1][2], 'oor': result[1][3], 'oor + mask': result[1][4],
                           'rpart': result[1][5], 'rpart + mask': result[1][6], 'ctree': result[1][7],
                           'ctree + mask': result[1][8]})
    # print(output)
    # pickle the entire array of results
    with open('r2_values.pkl', 'wb') as file:
        pickle.dump(output, file)

    print('success')
    # # code below is for testing purposes
    # np.random.seed(0)
    # DGP = 'quadratic'
    # missing_mechanism = 'MCAR'
    # model_type = 'ctree'
    # mask = False
    # d = 9
    # size = 2000
    # """
    # mean imputation
    # """
    # data_train = generate_data(size=size, d=d, imputation_value='nan', DGP=DGP, missing=missing_mechanism)
    # print(data_train)
    
    # model = train_model(data_train, d=d, DGP=DGP, model=model_type, mask=mask)
    # print(model)

    # data_test = generate_data(size=size, d=d, imputation_value='nan', DGP=DGP, missing=missing_mechanism)

    # predictions = make_predictions(data_test, d, model_type, mask, model)

    # print(r2_score(data_test['Y'], predictions))

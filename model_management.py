"""
Code that trains different kinds of functions given a dataset and also makes predictions
given a model and testing data.

Created: January 15, 2024

Updated: January 16, 2024
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

# for testing purposes, import code for generating a dataset
from data_generation import generate_data
from data_generation import create_matrix

# Import necessary packages to use R in Python
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula

def create_formula(data, mask):
    # helper function for creating a regression formula

    vars = data.columns

    # create the formula
    formula = 'Y~'
    for i in range(len(vars)):
        if vars[i][0] == 'R':
            if mask:
                formula += '+'+vars[i]
        elif vars[i][0] == 'Y':
            continue
        else:
            formula += '+'+vars[i]

    return formula

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

        formula = create_formula(data, mask)

        # train a decision tree, and use all default values as specified in Josse et al.
        tree = rpart_package.rpart(formula=Formula(formula), data=data)
        # print(tree)

        return tree
    elif model == 'ctree':
        # import the partykit package from R which contains the ctree implementation
        partykit_package = importr('partykit')

        formula = create_formula(data, mask)
        
        # train a ctree using all default values
        # ctree(y~., data=train, controls=ctree_control(
        #            minbucket=min_samples_leaf, mincriterion=0.0)), min_samples_leaf = 30
        # set min_samples_leaf
        tree = partykit_package.ctree(formula=Formula(formula), control=partykit_package.ctree_control(minbucket=30, mincriterion=0.0), data=data)

        return tree
    elif model == 'random_forest':
        # import the ranger package from R which contains the random forest implementation
        ranger_package = importr('ranger')

        formula = create_formula(data, mask)

        # train a random forest using all default values from the package
        rand_forest = ranger_package.ranger(formula=Formula(formula), data=data)

        # return the model
        return rand_forest
    elif model == 'xgboost':
        # first step is to recombine all the X variables and/or the R variables into a single
        # two-dimensional array in the original format of the multivariate normal function

        matrix = create_matrix(data, mask)

        # change the input format
        train_matrix = xgb.DMatrix(matrix, label=data['Y'])

        # train the model
        xgboost_model = xgb.train({}, train_matrix)

        return xgboost_model


def make_predictions(data_test, d, model_type, mask, model, num_missing=3):
    """
    Given a trained model and testing data, make predictions for the outcome
    variable.
    """
    # create an empty array as the return value
    predictions = []

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
    elif model_type == 'random_forest':
        predictions = robjects.r.predict(model, data=data_test)
        # in the ranger package, the first value of the predictions contains the
        # actual predicted values
        predictions = list(predictions[0])
    elif model_type == 'xgboost':
        # first step is to recombine all the X variables and/or the R variables into a single
        # two-dimensional array

        matrix = create_matrix(data_test, mask)

        # change the input format 
        test_matrix = xgb.DMatrix(matrix)

        # use the model to make predictions
        predictions = model.predict(test_matrix)

    return predictions

if __name__ == "__main__":
    np.random.seed(0)

    data = generate_data()
    print(data)
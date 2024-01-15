import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

# Import necessary packages to use R in Python
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula

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
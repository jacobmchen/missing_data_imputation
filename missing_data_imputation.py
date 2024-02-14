"""
Simulations to test if imputing by a constant is sufficient to train a supervised
classifier according to paper by Julie et al.

Created: November 9, 2023

Last updated: January 16, 2024

To-do:
1. statistical validation
2. discussion on computational complexity (can just time how long the method takes)

paired t-test for statistical validation, requirement is that they must be from the same sample

R^2 ~ method + experiment (one-hot encode the values for method and experiment)

___________
table: nrow= number of expérience
col = number of methods
ttest paired
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import pickle
from data_generation import generate_data
from model_management import train_model
from model_management import make_predictions
import sys

from rpy2.robjects import pandas2ri

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
    predictions = list(pd.Series(predictions).fillna(np.mean(predictions)))
    return r2_score(data_test['Y'], predictions)
    
def run_experiments(repetitions=1000, model_type='rpart', missing_mechanism='MCAR', verbose=False):
    """
    This function runs the synthetic experiments using four scenarios:
    1. no missing values as a baseline
    2. dropping rows of missing values
    3. imputing by the means of the missing variables
    4. imputing by some out of range value

    It repeats each scenario by the specified amount of times and reports
    the average of the R^2 values.
    """
    if model_type == 'rpart':
        num_methods = 12
        r2_values = [[], [], [], [], [], [], [], [], [], [], [], []]
    else:
        num_methods = 8
        r2_values = [[], [], [], [], [], [], [], []]

    size = 2000
    np.random.seed(0)

    # model_type = 'rpart'
    DGP = 'quadratic'
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
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, [999999]*d))
        cnt += 1

        """
        out of range imputation with mask
        """
        mask = True
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, [999999]*d))
        cnt += 1

        """
        CART with surrogate splits with no mask
        """
        if model_type == 'rpart':
            mask = False
            r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'nan'))
            cnt += 1

        """
        CART with surrogate splits with mask
        """
        if model_type == 'rpart':
            mask = True
            r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'nan'))
            cnt += 1

        """
        ctree with no mask
        """
        if model_type == 'rpart':
            mask = False
            r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, 'ctree', 'nan'))
            cnt += 1

        """
        ctree with mask
        """
        if model_type == 'rpart':
            mask = True
            r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, 'ctree', 'nan'))
            cnt += 1

        """
        MIA
        """
        mask = False
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'mia'))
        cnt += 1

        """
        Gaussian
        """
        mask = False
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'gaussian'))
        cnt += 1

        """
        Gaussian + mask
        """
        mask = True
        r2_values[cnt].append(run_single_experiment(size, d, DGP, missing_mechanism, mask, model_type, 'gaussian'))
        cnt += 1

    return [np.mean(r2_values[i]) for i in range(num_methods)], r2_values

if __name__ == "__main__":
    error_message = 'incorrect usage: use \'python3 missing_data_imputation.py False\' or \'python3 missing_data_imputation.py True\' to indicate whether testing'

    # Must be activated to use R packages in Python
    pandas2ri.activate()
    np.random.seed(0)

    if len(sys.argv) >= 2:
        testing = sys.argv[1]

        if testing == 'True':
            print(run_single_experiment(1000, 10, 'quadratic', 'predictive', True, 'random_forest', 'mean'))
        elif testing == 'False':

            missing_mechanisms = ['MCAR', 'MNAR', 'predictive']
            model_types = ['rpart', 'random_forest', 'xgboost']

            for missing_mechanism in missing_mechanisms:
                for model_type in model_types:
                    result = run_experiments(repetitions=1000, model_type=model_type, missing_mechanism=missing_mechanism)
                    print('mean values', result[0])
                    
                    if model_type == 'rpart':
                        output = pd.DataFrame({'mean': result[1][1], 'mean + mask': result[1][2], 'oor': result[1][3], 'oor + mask': result[1][4],
                                        'rpart': result[1][5], 'rpart + mask': result[1][6], 'ctree': result[1][7],
                                        'ctree + mask': result[1][8], 'mia': result[1][9], 'gaussian': result[1][10],
                                        'gaussian + mask': result[1][11]})
                    else:
                        output = pd.DataFrame({'mean': result[1][1], 'mean + mask': result[1][2], 'oor': result[1][3], 'oor + mask': result[1][4],
                                        'mia': result[1][5], 'gaussian': result[1][6],
                                        'gaussian + mask': result[1][7]})
                    print(output.columns)

                    # pickle the entire array of results
                    with open('r2_values_'+missing_mechanism+'_'+model_type+'.pkl', 'wb') as file:
                        pickle.dump(output, file)

            print('success')
        else:
            print(error_message)
    else:
        print(error_message)

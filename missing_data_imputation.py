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
from data_generation import generate_data
from model_management import train_model
from model_management import make_predictions

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
    # predictions = list(pd.Series(predictions).fillna(np.mean(predictions)))

    return r2_score(data_test['Y'], predictions)
    
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

    testing = True

    if testing:
        np.random.seed(0)
        DGP = 'quadratic'
        missing_mechanism = 'MCAR'
        model_type = 'ctree'
        mask = False
        d = 9
        size = 2000
        """
        mean imputation
        """
        data_train = generate_data(size=size, d=d, imputation_value='nan', DGP=DGP, missing=missing_mechanism)
        print(data_train)
        
        model = train_model(data_train, d=d, DGP=DGP, model=model_type, mask=mask)
        print(model)

        data_test = generate_data(size=size, d=d, imputation_value='nan', DGP=DGP, missing=missing_mechanism)

        predictions = make_predictions(data_test, d, model_type, mask, model)

        print(r2_score(data_test['Y'], predictions))
    else:
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

import numpy as np
import pandas as pd
import statsmodels.api as sm

def em_imputation(data, data_matrix, mu_hat, sigma_hat, d, num_missing):
    """
    Function that performs imputation using the EM algorithm. The input to this function
    is a pandas dataframe, a matrix representation of the dataframe, the estimate mu_hat
    from the EM algorithm, the estimate sigma_hat from the EM algorithm, the number of variables,
    and the number of variables that are partially observed.

    This function returns a pandas dataframe with the missing values imputed using the 
    """
    # normalization operation for sigma_hat
    sigma_hat = (0.99 * sigma_hat) + (np.trace(sigma_hat) * np.identity(d))

    for i in range(len(data_matrix)):
        # a value of True means drop that element
        # arrays where value of True means drop that element and False means keep that
        # element
        drop_observed = []
        drop_missing = []
        for j in range(num_missing):
            if data.at[i, 'R'+str(j+1)] == 0:
                drop_observed.append(False)
                drop_missing.append(True)
            else:
                drop_observed.append(True)
                drop_missing.append(False)
        for j in range(d-num_missing):
            drop_observed.append(True)
            drop_missing.append(False)

        # step 1: \mu_{mis(m)} vector
        # drop the observed variables
        mu_mis = np.delete(mu_hat, drop_observed, 0)

        # step 2: \sigma_{mis(m), obs(m)} matrix
        # rows: drop the observed rows
        # columns: drop the missing columns
        sigma_mis_obs = np.delete(sigma_hat, drop_observed, 0)
        sigma_mis_obs = np.delete(sigma_mis_obs, drop_missing, 1)

        # step 3: \sigma_{obs(m)} matrix
        # rows: drop the missing rows
        # columns: drop the missing columns
        sigma_obs = np.delete(sigma_hat, drop_missing, 0)
        sigma_obs = np.delete(sigma_obs, drop_missing, 1)

        # step 4: X_{obs(m)} vector
        # actual values of observed variables, drop the missing rows of missing data
        X_obs = np.delete(data_matrix[i], drop_missing, 0)

        # step 5: \mu_{obs(m)} vector
        # drop the missing variables
        mu_obs = np.delete(mu_hat, drop_missing, 0)

        # step 6: apply the formula to get the imputed missing values
        impute_values = mu_mis + (sigma_mis_obs @ np.linalg.inv(sigma_obs) @ (X_obs - mu_obs))
        
        # put the imputed values into the data matrix
        cnt = 0
        for j in range(len(data_matrix[i])):
            if np.isnan(data_matrix[i][j]):
                data_matrix[i][j] = impute_values[cnt]
                cnt += 1

    # X = []
    for i in range(d):
        # X.append(np.array(data_matrix[:, i]))
        X = np.array(data_matrix)[:, i]
        data['X'+str(i+1)] = X
    
    return data
    
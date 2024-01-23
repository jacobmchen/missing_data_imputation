import pandas as pd
import numpy as np
from scipy import stats
import pickle

with open('r2_values_MCAR.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)
alpha = 0.01
significantly_different = []

# perform a pairwise test
for i in range(len(data.columns)):
    for j in range(i+1, len(data.columns)):
        print('paired t-test:', data.columns[i], ',', data.columns[j])
        print(stats.ttest_rel(data[data.columns[i]], data[data.columns[j]])[1])
        if stats.ttest_rel(data[data.columns[i]], data[data.columns[j]])[1] <= alpha:
            significantly_different.append((data.columns[i], data.columns[j]))

print(len(significantly_different))
print(significantly_different)

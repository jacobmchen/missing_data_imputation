import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
Code to plot experimental results from missing_data_imputation.py.
"""

with open('r2_values_MNAR.pkl', 'rb') as file:
    data = pickle.load(file)

fig, ax = plt.subplots()

# build a box plot
ax.boxplot(data, vert=False)

# title and axis labels
ax.set_title('MNAR, Decision Tree')
ax.set_xlabel('R^2')
ax.set_ylabel('Imputation Strategy/Model')
labels = data.columns
yticklabels = list(labels)
ax.set_yticklabels(yticklabels)

# add horizontal grid lines
ax.xaxis.grid(True)

# show the plot
plt.show()

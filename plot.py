import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

"""
Code to plot experimental results from missing_data_imputation.py.
"""

if __name__ == "__main__":
    missing_mechanism = ['MCAR', 'MNAR', 'predictive']
    model_type = ['rpart', 'random_forest', 'xgboost']

    fig, ax = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            with open('r2_values_'+missing_mechanism[j]+'_'+model_type[i]+'.pkl', 'rb') as file:
                data = pickle.load(file)

            # build a box plot
            ax[i, j].boxplot(data, whis=(0,100), vert=False)

            # title and axis labels
            ax[i, j].set_title(missing_mechanism[j]+'_'+model_type[i], fontsize=6)
            # ax[0, 0].set_xlabel('R^2')
            # ax[0, 0].set_ylabel('Imputation Strategy/Model')

            if j == 0:
                labels = data.columns
            else:
                labels = ['']*len(data.columns)
            yticklabels = list(labels)
            ax[i, j].set_yticklabels(yticklabels)

            # add horizontal grid lines
            ax[i, j].xaxis.grid(True)

    # show the plot
    plt.show()

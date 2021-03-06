# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:02:28 2019

@author: LENOVO
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import r2_score

df = pd.read_csv("forestfires.csv")

#########################
# 1. DATA PREPROCESSING #
#########################

# delete day of week information - don't expect it to be relevant


# extract feature and target variables from dataframe
X = df.iloc[:, 0:10].values    
y= df.iloc[:, 10].values

# split training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=0)

############################################
# 2. MODEL AND TEST TRAINING DATA WITH SVM #
############################################

# correlation matrix 
cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
firenum_map_mat = np.full((9, 9), 0.0)
firearea_map_mat = np.full((9, 9), 0.0)
firedensity = np.full((9, 9), 0.0)

for row in range(0, df.shape[0]):
    x = df['X'][row]-1
    y = df['Y'][row]-1
    firenum_map_mat[x, y] += 1
    firearea_map_mat[x, y] += df['area'][row]

for (x, y), value in np.ndenumerate(firenum_map_mat):
    if (firenum_map_mat[x, y] != 0):
        firedensity[x, y] = firearea_map_mat[x, y]/firenum_map_mat[x, y]

sns.set(font_scale=1.5)
heat_map = sns.clustermap(firenum_map_mat,
                       cbar=True,
                       annot=True,
                       square=True,
                       fmt='.2f',
                       annot_kws={'size': 12},
                       yticklabels=cols,
                       xticklabels=cols)
plt.title('Fire Numbers on Park Zones')
plt.show()

sns.set(font_scale=1.5)
heat_map = sns.clustermap(firearea_map_mat,
                       cbar=True,
                       annot=True,
                       square=True,
                       fmt='.2f',
                       annot_kws={'size': 12},
                       yticklabels=cols,
                       xticklabels=cols)
plt.title('Fire Areas on Park Zones')
plt.show()

sns.set(font_scale=1.5)
heat_map = sns.heatmap(firedensity,
                       cbar=True,
                       annot=True,
                       square=True,
                       fmt='.2f',
                       annot_kws={'size': 12},
                       yticklabels=cols,
                       xticklabels=cols)
plt.title('Fire Areas on Park Zones')
plt.show()
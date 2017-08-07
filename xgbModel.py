# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

import os
os.chdir('C:\\Users\\Ran\\Documents\\My Projects\\Capital Bike Web')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_pickle('dataset_out_MF.pkl')
temp = []
for x in dataset['weekday']:
    if x < 6:
        temp.append(1)
    else:
        temp.append(0)
dataset['weekday'] = temp

DemandByStation = pd.read_pickle('DemandByStation.pkl')
totalcount_wd = DemandByStation[DemandByStation['weekday'] == 1]
totalcount_wd['n'] = totalcount_wd['n']/5
totalcount_we = DemandByStation[DemandByStation['weekday'] == 0]
totalcount_we['n'] = totalcount_we['n']/2
totalcount_ave = totalcount_wd.append(totalcount_we)
#totalcount_ave.to_pickle('totalcount_ave.pkl')
totalcount = DemandByStation.groupby(['start station'])['n'].sum()
#totalcount.to_pickle('totalcount.pkl')
totalcount.sort_values(ascending = False, inplace = True)
topstation = list(totalcount.index)

#select top station dataset to predict
N = 0
station = topstation[N]
datasetN = dataset[(dataset['start station'] == station)]
datasetWday = datasetN[(datasetN['weekday'] == 1)]
datasetWend = datasetN[(datasetN['weekday'] == 0)]


#select the x and y
Xday = datasetWday.iloc[:, [1,3,5]].values
yday = datasetWday.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xday, yday, test_size = 0.2, random_state = 0)


# Fitting XGBoost to the Training set
import xgboost as xgb
##
#  this script demonstrate how to fit generalized linear model in xgboost
#  basically, we are using linear model, instead of tree for our boosters
##
xgb_model = xgb.XGBRegressor().fit(X_train,y_train)
y_pred_xgb = xgb_model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
R2 = r2_score(y_test, y_pred_xgb)
MSE = mean_squared_error(y_test, y_pred_xgb)
print(MSE, R2)

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = xgb_model, X = X_train, y = y_train, scoring='neg_mean_squared_error', cv = 10)
print(score.mean())
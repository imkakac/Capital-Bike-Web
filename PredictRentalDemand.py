# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 00:16:42 2017

@author: Ran
"""

#predicting DC bike share demand
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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# Fitting Polynomial Regression to the train dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
n3 = 15
X_poly = PolynomialFeatures(n3).fit_transform(X_train)
# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
X_prepoly = PolynomialFeatures(n3).fit_transform(X_test)
y_pred = lin_reg.predict(X_prepoly)
#y_pred = sc_y.inverse_transform(y_pred)
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_train, lin_reg.predict(X_poly))
print(MSE, R2)

# Applying k-Fold Cross Validation
'''from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = lin_reg, X = X_poly, y = y_train, scoring='neg_mean_squared_error', cv = 10)
print(score.mean())'''

# Visualising the Linear Regression results
from datetime import date
d = date(2016, 7, 11) #(year, month, day)
d1 = d.isocalendar()
hour = range(0,24)
year = d1[0]
month = d.month
weeknum = d1[1]
weekday = d1[2]

truedata_plot = datasetWday.iloc[:, [5,6]][(datasetWday['year'] == year) & (datasetWday['weeknumber'] == weeknum)]
X_plot0 = np.array([[year, weeknum],]*24)
X_plot = np.insert(X_plot0, 2, hour, axis=1)
#scale
X_plot = sc_X.transform(X_plot)
Xplot_poly = PolynomialFeatures(n3).fit_transform(X_plot)
ypre_plot = lin_reg.predict(Xplot_poly)
#ypre_plot = sc_y.inverse_transform(ypre_plot)

plt.scatter(truedata_plot.iloc[:,0], truedata_plot.iloc[:,1], color = 'red')
plt.plot(hour, ypre_plot, color = 'blue')
plt.title('Station '+station+' Weekday: '+str(d)+' Week '+str(weeknum))
plt.xlabel('Hour')
plt.ylabel('Rental Count')
plt.show()

#########   Weekend   ############
#select the x and y
Xend = datasetWend.iloc[:, [1,3,5]].values
yend = datasetWend.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xend, yend, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Polynomial Regression to the train dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
n3 = 16
X_poly = PolynomialFeatures(n3).fit_transform(X_train)
# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
X_prepoly = PolynomialFeatures(n3).fit_transform(X_test)
y_pred = lin_reg.predict(X_prepoly)
#y_pred = sc_y.inverse_transform(y_pred)
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_train, lin_reg.predict(X_poly))
print(MSE, R2)
# Applying k-Fold Cross Validation
'''from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = lin_reg, X = X_poly, y = y_train, scoring='neg_mean_squared_error', cv = 10)
print(score.mean())'''

# Visualising the Linear Regression results
from datetime import date
d = date(2016, 6, 11) #(year, month, day)
d1 = d.isocalendar()
hour = range(0,24)
year = d1[0]
month = d.month
weeknum = d1[1]
weekday = d1[2]

truedata_plot = datasetWend.iloc[:, [5,6]][(datasetWend['year'] == year) & (datasetWend['weeknumber'] == weeknum)]
X_plot0 = np.array([[year, weeknum],]*24)
X_plot = np.insert(X_plot0, 2, hour, axis=1)
#scale
X_plot = sc_X.transform(X_plot)
Xplot_poly = PolynomialFeatures(n3).fit_transform(X_plot)
ypre_plot = lin_reg.predict(Xplot_poly)
#ypre_plot = sc_y.inverse_transform(ypre_plot)

plt.scatter(truedata_plot.iloc[:,0], truedata_plot.iloc[:,1], color = 'red')
plt.plot(hour, ypre_plot, color = 'blue')
plt.title('Station '+station+' Weekend: '+str(d)+' Week '+str(weeknum))
plt.xlabel('Hour')
plt.ylabel('Rental Count')
plt.show()
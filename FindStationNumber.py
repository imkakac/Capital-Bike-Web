# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:31:16 2017
Find station number for station address
@author: Ran
"""


#history_data[5:19][3:5]
#def findStationNumber(data, StationInfo):
def findStationNumber():
    import pandas as pd
#    history_data = data
#    Station_Info_Final = StationInfo
    for i in range(5,19):
        teststring = history_data[i].iloc[:,3:5]
        teststring = pd.merge(teststring, Station_Info_Final, how = 'left', left_on = 'start station', right_on = 'station address', left_index = True)
        teststring = pd.merge(teststring, Station_Info_Final, how = 'left', left_on = 'end station', right_on = 'station address')
        teststring = teststring.drop(['start station','end station','station address_x', 'station address_y'], axis=1)
        history_data[i].loc[:,3:5] = teststring.values
    return history_data


'''
import timeit
i=6
history_data[i] = pd.read_csv(csv_list[i])
# drop row with nan
history_data[i]=history_data[i].dropna(how='any') 
history_data[i].drop(history_data[i].columns[5], 1, inplace=True)
history_data[5].rename(columns = {'Type': 'Member Type'}, inplace=True)
history_data[6].rename(columns = {'Bike Key': 'Member Type'}, inplace=True)
for i in range(7,13):
    history_data[i].rename(columns = {'Subscription Type': 'Member Type'}, inplace=True)
history_data[14].rename(columns = {'Subscriber Type': 'Member Type'}, inplace=True)
for i in range(15,18):
    history_data[i].rename(columns = {'Subscription Type': 'Member Type'}, inplace=True)
history_data[18].rename(columns = {'Subscription type': 'Member Type'}, inplace=True)
history_data[22].rename(columns = {'Account type': 'Member Type'}, inplace=True)

#rename column duration
history_data[17].rename(columns = {'Total duration (ms)': 'Duration'}, inplace=True)
for i in range(18,26):
    history_data[i].rename(columns = {'Duration (ms)': 'Duration'}, inplace=True)

#rename column start station # & end station #
for i in range(17,27):
    history_data[i].rename(columns = {'Start station number': 'Start station', 'End station number': 'End station'}, inplace=True)

#change all column names to lower case
for i in range(0,27):
    history_data[i].columns = [x.lower() for x in history_data[i].columns]

#re order the column
for i in range(12,19):
    history_data[i] = history_data[i][['duration', 'start date', 'end date', 'start station', 'end station',
       'member type']]
'''


i=5 # max 18
testdf = history_data[i]
testdf2 = testdf
teststring = testdf.iloc[:,3:5]
#testindex = Station_Info_Final.iloc[:,0].index(teststring.iloc[0,0])


'''
start_time = timeit.default_timer()
# code you want to evaluate
teststring = teststring.replace(Station_Info_Final.iloc[:,0].tolist(),Station_Info_Final.iloc[:,1].tolist())
elapsed = timeit.default_timer() - start_time
print(elapsed)
'''

teststring2 = testdf.iloc[:,3:5]
start_time = timeit.default_timer()
# code you want to evaluate
teststring2 = pd.merge(teststring2, Station_Info_Final, how = 'left', left_on = 'start station', right_on = 'station address', left_index = True)
teststring2 = pd.merge(teststring2, Station_Info_Final, how = 'left', left_on = 'end station', right_on = 'station address')
#check na row
narow = teststring2[teststring2.isnull().any(axis=1)]
print(len(narow)/len(teststring2)*100)
#teststring2['start station'] = [int(i) for i in teststring2['start station']]
a = list(teststring2['start station'].unique())
teststring3 = teststring2.drop(['start station','end station','station address_x', 'station address_y'], axis=1)
#teststring3 = teststring3.astype(int)
#teststring2.columns.values[2:4] = ['start station', 'end station']

#testdf2.loc[:,3:5] = teststring3.values
elapsed = timeit.default_timer() - start_time
print(elapsed)

a = list(history_data[0]['start station'].unique())
b = [int(x) for x in a]

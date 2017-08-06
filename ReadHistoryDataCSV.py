# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#clear all variables
#import sys
#sys.modules[__name__].__dict__.clear()

import csv
import pandas as pd
import math
import numpy as np

folder = 'history data'
#find all .csv file in the folder
import glob
csv_list = glob.glob(folder + '/*.csv')



#read all .csv files and save as a dictionary
history_data = {}
#history_data2 = {}
XML_Station_Info = {}
for i in range(0, len(csv_list)):
        #read history trip data from csv
        history_data[i] = pd.read_csv(csv_list[i])
        # drop row with nan
        history_data[i]=history_data[i].dropna(how='any') 
        if i < 19:
            #remove row 'Bike #'
            history_data[i].drop(history_data[i].columns[5], 1, inplace=True)
        else:
            #remove rows 'Start Station', 'end station', and 'Bike #'
            ss = history_data[i].iloc[:,3:5]
            ss.columns = ['station number', 'station address']
            XML_Station_Info[i-19]=ss.drop_duplicates()
            history_data[i].drop(history_data[i].columns[[4,6,7]], 1, inplace=True)

XML_Station_Info = pd.concat(XML_Station_Info.values(), ignore_index=True)
XML_Station_Info = XML_Station_Info.drop_duplicates()
XML_Station_Info.columns = ['station number', 'station address']
XML_Station_Info = XML_Station_Info[['station address', 'station number']]

import time
import datetime
XML_Station_Info2 = {}
for i in range(0,5):
    temp = history_data[i]
    func = lambda x: x.split("(", 1)[1].split(")", 1)[0]
    func2 = lambda x: x.split(" (", 1)[0]
#    columns = ['Start station', 'End station']
    #index = np.arange(103) # array of numbers for the number of samples
#    teststring2 = pd.DataFrame(columns=columns)
    station_address = temp['Start station'].apply(func2)
    temp['Start station'] = temp['Start station'].apply(func)
    temp['End station'] = temp['End station'].apply(func)
    temp2 = pd.DataFrame(data = {'station address': station_address,'station number': temp['Start station']})
    temp2 = temp2.drop_duplicates()
    XML_Station_Info2[i] = temp2[['station address','station number']]
    history_data[i] = temp

XML_Station_Info2 = pd.concat(XML_Station_Info2.values(), ignore_index=True)
XML_Station_Info2 = XML_Station_Info2.drop_duplicates()
XML_Station_Info2['station number'][106] = 31234
XML_Station_Info2['station number'] = [int(i) for i in XML_Station_Info2['station number']]

from ReadXML import readStationXML
StationInfo = readStationXML()

Station_Info_Final = StationInfo.iloc[:,0:2].append(XML_Station_Info, ignore_index = True).append(XML_Station_Info2, ignore_index = True)  
#Station_Info_Final = Station_Info_Final.drop(Station_Info_Final.index[1044])
Station_Info_Final = Station_Info_Final.drop_duplicates()
#duplicated station address Calvert St & Woodley Pl NW keep 31121 remove 31106
Station_Info_Final.drop(Station_Info_Final.index[491], inplace = True)
Station_Info_Final = Station_Info_Final.reset_index(drop=True)

#Manually add station to the list
Station_Info_Final.loc[len(Station_Info_Final)] = ['New Hampshire Ave & T St NW [formerly 16th & U St NW]', int(31229)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Idaho Ave & Newark St NW [on 2nd District patio]', int(31302)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['4th St & Rhode Island Ave NE', int(31500)] 
Station_Info_Final.loc[len(Station_Info_Final)] = ['Virginia Square', int(31024)] 
Station_Info_Final.loc[len(Station_Info_Final)] = ['Central Library', int(31025)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Fairfax Dr & Glebe Rd', int(31038)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Pentagon City Metro / 12th & Hayes St', int(31005)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['8th & F St NW / National Portrait Gallery', int(31232)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['11th & K St NW', int(31263)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['McPherson Square / 14th & H St NW', int(31216)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['13th & U St NW', int(31268)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['S Abingdon St & 36th St S', int(31064)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['18th & Wyoming Ave NW', int(31114)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Connecticut Ave & Nebraska Ave NW', int(31310)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['12th & Hayes St /  Pentagon City Metro', int(31005)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Alta Tech Office', int(32002)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Bethesda Ave & Arlington Blvd', int(32002)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Fallsgove Dr & W Montgomery Ave', int(32016)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['Thomas Jefferson Cmty Ctr / 2nd St S & Ivy', int(31074)]
Station_Info_Final.loc[len(Station_Info_Final)] = ['22nd & Eads St', int(31013)]

    #Station_Info_Final = Station_Info_Final.append(['New Hampshire Ave & T St NW [formerly 16th & U St NW]',31229])

#Rename the column names in history data and re-orgnize
#rename column member type
history_data[5].rename(columns = {'Type': 'Member Type'}, inplace=True)
history_data[6].rename(columns = {'Bike Key': 'Member Type'}, inplace=True)
history_data[7].rename(columns = {'Subscriber Type': 'Member Type'}, inplace=True)
for i in range(8,13):
    history_data[i].rename(columns = {'Subscription Type': 'Member Type'}, inplace=True)
history_data[10].rename(columns = {'Start time': 'Start date'}, inplace=True)
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

#find station number
def findStationNumber():
#    import pandas as pd
#    history_data = data
#    Station_Info_Final = StationInfo
    for i in range(5,19):
        teststring = history_data[i].iloc[:,3:5]
        teststring = pd.merge(teststring, Station_Info_Final, how = 'left', left_on = 'start station', right_on = 'station address', left_index = True)
        teststring = pd.merge(teststring, Station_Info_Final, how = 'left', left_on = 'end station', right_on = 'station address')
        teststring = teststring.drop(['start station','end station','station address_x', 'station address_y'], axis=1)
        history_data[i].loc[:,3:5] = teststring.values
    return history_data

history_data = findStationNumber()

#dictionary to df
dataset = pd.concat(history_data.values(), ignore_index=True)

# convert time to year, month, day, weekday, hour
#from datetime import datetime
testlist = dataset.iloc[1:100,1]
#datetime_object = datetime.strptime(testlist, '%b/%d/%Y %H:%M')
datetime_object1 = pd.to_datetime(history_data[0].iloc[:,1])
datetime_object1[datetime_object1.isnull().any(axis=1)]

starttime_dict = {}
for i in range(0,27):
    starttime_dict[i] = pd.to_datetime(history_data[i].iloc[:,1])
    print(i)

for i in range(0,27):
    print(len(starttime_dict[i][starttime_dict[i].isnull()]))
    
#datetime_df = []
starttime_series = pd.concat(starttime_dict.values(), ignore_index=True)
datetime_table = [list(x.isocalendar())+[x.month, x.hour] for x in starttime_series]
datetime_df = pd.DataFrame(datetime_table, columns = ['year', 'weeknumber', 'weekday','month','hour'])

#Merge csv dataset and time dataset
final_dataset = dataset.join(datetime_df)
#save final dataset
final_dataset.to_pickle('final_dataset.pkl')
##create dataset for rental check out prediction
#dataset_out = final_dataset[['start station','year','month','weeknumber', 'weekday','hour']]
#dataset_out1 = dataset_out.groupby(['start station','year','month','weeknumber', 'weekday','hour']).count
#for i in range(0,len(dataset_out)):
#    print(i/len(dataset_out))
#    if dataset_out.loc['weekday',i] < 6:
#        dataset_out.loc['weekday',i] = 1
#    else:
#        dataset_out.loc['weekday',i] = 0

'''
aaa = history_data[0]
datetimestring = history_data[0].iloc[0,1]
timestamp = time.mktime(time.strptime(datetimestring, '%m/%d/%Y %H:%M'))
datetime.datetime.strptime(datetimestring, '%m/%d/%Y %H:%M').strftime('%A')
timeobject = datetime.datetime.strptime(datetimestring, '%m/%d/%Y %H:%M')


time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
time.strftime('%Y', time.localtime(timestamp))
time.strftime('%A', time.localtime(time.time()))
time.gmtime(timestamp)
time.gmtime(time.time())
'''

#create a function to extract the station number for history_data[0:2,4]


'''
#save the dictionary into .json
# -*- coding: utf-8 -*-
import json

# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Write JSON file
with io.open('data.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(history_data,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(to_unicode(str_))

# Read JSON file
with open('data.json') as data_file:
    data_loaded = json.load(data_file)

#print(data == data_loaded)
#cb_data = pd.read_csv(folder + '/2017-Q1-Trips-History-Data.csv')
'''
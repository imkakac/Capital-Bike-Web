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

folder = 'history data'
#find all .csv file in the folder
import glob
csv_list = glob.glob(folder + '/*.csv')

#read all .csv files and save as a dictionary
history_data = {}
#history_data2 = {}
for i in range(0, len(csv_list)):
        #read history trip data from csv
        history_data[i] = pd.read_csv(csv_list[i])
        if i < 19:
            #remove row 'Bike #'
            history_data[i].drop(history_data[i].columns[5], 1, inplace=True)
        else:
            #remove rows 'Start Station', 'end station', and 'Bike #'
            history_data[i].drop(history_data[i].columns[[4,6,7]], 1, inplace=True)
            
import time

aaa = history_data[0]
datetimestring = history_data[0].iloc[0,1]
timestamp = time.mktime(time.strptime(datetimestring, '%m/%d/%Y %H:%M'))
datetime.datetime.strptime(datetimestring, '%m/%d/%Y %H:%M').strftime('%A')
timeobject = datetime.datetime.strptime(datetimestring, '%m/%d/%Y %H:%M')

import datetime
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
time.strftime('%Y', time.localtime(timestamp))
time.strftime('%A', time.localtime(time.time()))
time.gmtime(timestamp)
time.gmtime(time.time())

#testdf = history_data[0].iloc[0:10,0:2]
#testdf['new_date']

#history_data_Testdf = history_data[26].drop(history_data[26].columns[[4,6,7]],1)
#history_data_Testdf2 = aaa.drop(aaa.columns[5],1)
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
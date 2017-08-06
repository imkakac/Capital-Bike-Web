# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:35:02 2017

@author: Ran
"""

#create dataset_out
#load final dataset
import pandas as pd
final_dataset = pd.read_pickle('final_dataset.pkl')

#create dataset for rental check out prediction
dataset_out = final_dataset[['start station','year','month','weeknumber', 'weekday','hour']]
import timeit
#testdata = dataset_out.iloc[0:10000,:]

start_time = timeit.default_timer()
dataset_out1 = dataset_out.groupby(['start station','year','month','weeknumber', 'weekday','hour'])['hour'].count()
a = list(dataset_out1.index)
dataset_out_table1 = [list(x) for x in a]
dataset_out_df1 = pd.DataFrame(dataset_out_table1,columns = ['start station','year','month','weeknumber', 'weekday','hour']) 
dataset_out1 = dataset_out1.reset_index(drop=True)
dataset_out1.rename('n',inplace = True)
dataset_out_df = dataset_out_df1.join(dataset_out1)
elapsed = timeit.default_timer() - start_time
print(elapsed)
dataset_out_df.to_pickle('dataset_out_MF.pkl')

#convert monday - sunday to weekday(1) or weekend(0)
start_time = timeit.default_timer()

#temp = [1 for x in dataset_out_df2['weekday'] if x < 6]
#testdf = dataset_out_df.iloc[0:1000,:]
temp = []
for x in dataset_out_df['weekday']:
    if x < 6:
        temp.append(1)
    else:
        temp.append(0)
elapsed = timeit.default_timer() - start_time
print(elapsed)

#conunt n by weekday or weekend
start_time = timeit.default_timer()
dataset_out_df['weekday'] = temp
dataset_out2 = dataset_out_df.groupby(['start station','year','month','weeknumber', 'weekday','hour'])['n'].count()
b = list(dataset_out2.index)
dataset_out_table2 = [list(x) for x in b]
dataset_out_df2 = pd.DataFrame(dataset_out_table2,columns = ['start station','year','month','weeknumber', 'weekday','hour']) 
dataset_out2 = dataset_out2.reset_index(drop=True)
dataset_out2.rename('n',inplace = True)
dataset_out_df3 = dataset_out_df2.join(dataset_out2)
elapsed = timeit.default_timer() - start_time
print(elapsed)
dataset_out_df3.to_pickle('dataset_out_WD.pkl')
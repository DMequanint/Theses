#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import time
import datetime
df1= pd.read_csv('C:/2020-2021/Data/Datasets/Nimda/dataset_nimda_513_1_rrc04_balancedF.csv', delimiter=',')
df2= pd.read_csv('C:/2020-2021/Data/Datasets/Slammer/dataset_slammer_513_1_rrc04_balancedF.csv', delimiter=',')
df3= pd.read_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_1_rrc04_balancedF.csv', delimiter=',')
df4 = pd.read_csv('C:/2020-2021/Data/Datasets/AWSLeak/dataset_aws-leak_15547_1_rrc04_balancedF.csv', delimiter=',')

df1["eventName"] = df1.groupby(df1["class"]).grouper.group_info[0]
df1.loc[df1['eventName'] == 0, 'eventName'] = 'Regular'
df1.loc[df1['eventName'] == 1, 'eventName'] = 'Indirect Anomaly'
df1["activityName"] = df1.groupby(df1["class"]).grouper.group_info[0]
df1.loc[df1['activityName'] == 0, 'activityName'] = 'Normal'
df1.loc[df1['activityName'] == 1, 'activityName'] = 'Nimda vrius Attack'
first_column = df1.pop('eventName')
df1.insert(0, 'eventName', first_column)
second_column = df1.pop('activityName')
df1.insert(1, 'activityName', second_column)

df2["eventName"] = df2.groupby(df2["class"]).grouper.group_info[0]
df2.loc[df2['eventName'] == 0, 'eventName'] = 'Regular'
df2.loc[df2['eventName'] == 1, 'eventName'] = 'Indirect Anomaly'
df2["activityName"] = df2.groupby(df2["class"]).grouper.group_info[0]
df2.loc[df2['activityName'] == 0, 'activityName'] = 'Normal'
df2.loc[df2['activityName'] == 1, 'activityName'] = 'Slammer vrius Attack'
first_column = df2.pop('eventName')
df2.insert(0, 'eventName', first_column)
second_column = df2.pop('activityName')
df2.insert(1, 'activityName', second_column)

df3["eventName"] = df3.groupby(df3["class"]).grouper.group_info[0]
df3.loc[df3['eventName'] == 0, 'eventName'] = 'Regular'
df3.loc[df3['eventName'] == 1, 'eventName'] = 'Direct Unintended Anomaly'
df3["activityName"] = df3.groupby(df3["class"]).grouper.group_info[0]
df3.loc[df3['activityName'] == 0, 'activityName'] = 'Normal'
df3.loc[df3['activityName'] == 1, 'activityName'] = 'Telecom Malesian leak'

first_column = df3.pop('eventName')
df3.insert(0, 'eventName', first_column)
second_column = df3.pop('activityName')
df3.insert(1, 'activityName', second_column)

df4["eventName"] = df4.groupby(df4["class"]).grouper.group_info[0]
df4.loc[df4['eventName'] == 0, 'eventName'] = 'Regular'
df4.loc[df4['eventName'] == 1, 'eventName'] = 'Direct Unintended Anomaly'
df4["activityName"] = df4.groupby(df4["class"]).grouper.group_info[0]
df4.loc[df4['activityName'] == 0, 'activityName'] = 'Normal'
df4.loc[df4['activityName'] == 1, 'activityName'] = 'AWS Leak'

first_column = df4.pop('eventName')
df4.insert(0, 'eventName', first_column)
second_column = df4.pop('activityName')
df4.insert(1, 'activityName', second_column)

frames = [df1, df2, df3, df4]
df5= pd.concat(frames)
df5.to_csv('C:/2020-2021/Data/Datasets/CombinedDataset/dataset_combined_1_rrc04_balancedF.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





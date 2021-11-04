#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import datetime
df = pd.DataFrame( columns=["announcements", "nlri_ann", "withdrawals", "origin_changes", "number_rare_ases", "as_path_max", "imp_wd", "rare_ases_max",
                        "as_path_avg", "nadas", "flaps", "dups", "news", "timestamp"])
dfsource = pd.read_csv('C:/2020-2021/Data/Datasets/dataset_malaysian-telecom_513_5_rrc04.csv', delimiter=',')
df["announcements"]=dfsource["announcements"]
df["nlri_ann"]=dfsource["nlri_ann"]
df["withdrawals"]=dfsource["withdrawals"]
df["origin_changes"]= dfsource["origin_changes"]
df["number_rare_ases"]=dfsource["number_rare_ases"]
df["as_path_max"]=dfsource["as_path_max"]
df["imp_wd"]=dfsource["imp_wd"]
df["rare_ases_max"]=dfsource["rare_ases_max"]
df["as_path_avg"]=dfsource["as_path_avg"]
df["nadas"]=dfsource["nadas"]
df["flaps"]=dfsource["flaps"]
df["dups"]=dfsource["dups"]
df["news"]=dfsource["news"]
#df["timestamp"]=dfsource["timestamp"] 
df["timestamp"] = pd.to_datetime(dfsource["timestamp"])
df["class"]= dfsource["class"]
#df["timestamp:year"] = df["timestamp"].dt.year
#df["timestamp:month"] = df["timestamp"].dt.month
#df["timestamp:day"] = df["timestamp"].dt.day
#df["caseID"] = df.groupby(df.timestamp.dt.floor('15T')).ffill(axis = 1)
#df["caseID"]= pd.Series(df.groupby(pd.Grouper(freq='15T',key='event_date')))
df["caseID"]= df.groupby(df.timestamp.dt.floor('15T')).grouper.group_info[0] + 1
df["caseID"] = "M0" + df["caseID"].astype(str) 
df["eventType"] = df.groupby(df["class"]).grouper.group_info[0]
df.loc[df['eventType'] == 0, 'eventType'] = 'Regular'
df.loc[df['eventType'] == 1, 'eventType'] = 'Direct Unintended Anomaly'
df["activityName"] = df.groupby(df["class"]).grouper.group_info[0]
df.loc[df['activityName'] == 0, 'activityName'] = 'Normal'
df.loc[df['activityName'] == 1, 'activityName'] = 'Telecom Malesian leak'
first_column = df.pop('caseID')
df.insert(0, 'caseID', first_column)
second_column = df.pop('eventType')
df.insert(1, 'eventType', second_column)
third_column = df.pop('activityName')
df.insert(2, 'activityName', third_column)
df.drop("class", inplace=True, axis=1)
df.to_csv('C:/2020-2021/Data/Datasets/dataset_malaysian-telecom_513_5_rrc04_V2.csv', index=False)
df.head(100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





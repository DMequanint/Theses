#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import datetime
import numpy as np
import random
df = pd.DataFrame( columns=["announcements", "nlri_ann", "withdrawals", "origin_changes", "number_rare_ases", "as_path_max", "imp_wd", "rare_ases_max",
                        "as_path_avg", "nadas", "flaps", "dups", "news", "timestamp"])
dfsource = pd.read_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_1_rrc04_EqualF.csv', delimiter=',')
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
#df["caseID"]= df.groupby(df.timestamp.dt.floor('15T')).grouper.group_info[0] + 1
 
df["eventName"] = df.groupby(df["class"]).grouper.group_info[0]
df.loc[df['eventName'] == 0, 'eventName'] = 'Regular'
df.loc[df['eventName'] == 1, 'eventName'] = 'Direct Unintended Anomaly'
df["activityName"] = df.groupby(df["class"]).grouper.group_info[0]
df.loc[df['activityName'] == 0, 'activityName'] = 'Normal'
df.loc[df['activityName'] == 1, 'activityName'] = 'Telecom Malesian leak'

#grouped_df = df.groupby(["activityName", "origin_changes"])
df["eventType"] = df.groupby(["activityName", "origin_changes"]).grouper.group_info[0] + 1
#for key,item in grouped_df:['values'].nunique()
 #a_group = grouped_df.get_group(key)
df["eventType"] = "Evt0" + df["eventType"].astype(str)
#df["caseID"]=''
#df["caseID"]= df.groupby("eventType").grouper.group_info[0] + 1    
#grouped_df = df.groupby(["activityName", "origin_changes"])
#grouped_df = df.groupby(["eventType"])
#grouped_df=df.sample(frac = 1.0).groupby('eventType').head(1)
"""n=313
random.shuffle(df['eventType'])
groups=np.split(df['eventType'],range(n,len(df['eventType']),n))
dfr=df.set_index(df['eventType'])
for gp in groups : print (dfr.loc[gp])"""
#for key,item in grouped_df:
        #df["caseID"] = "CID0" + df["caseID"].astype(str)
        #print(key)
        #print(grouped_df)
        #df["caseID"] = "CID0" + grouped_df.get_group(key).astype(str)
#grouped_df.get_group('Evt021')
"""by_eventType = df.groupby("eventType")
for event, frame in by_eventType:
    print(frame.head(2))
    
    #by_eventType.groups("Evt021")  # Index locations for the rows
    #by_eventTpe.group_by("Evt021") # Basically equivalent to df.loc[df["eventType"]== "Evt021"]
event, frame = next(iter(by_eventType)) # First tuple from iterator
rows = df.shape[0]"""
#first_column = df.pop('caseID')
#df.insert(0, 'caseID', first_column)
first_column = df.pop('eventType')
df.insert(0, 'eventType', first_column)
second_column = df.pop('eventName')
df.insert(1, 'eventName', second_column)
third_column = df.pop('activityName')
df.insert(2, 'activityName', third_column)
df.drop("class", inplace=True, axis=1)
cnt=1
test = pd.DataFrame(columns=[df.columns])
#firstRound = False
df1 = df
dfselected = df
df1=df[df['origin_changes']>=0]
lensum=0
j=1
k=0
df5 = pd.DataFrame(columns=['caseID'])
df6 = pd.DataFrame(columns=['eventType'])
df.to_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_5_rrc04_V2.csv', index=False)
for name, values in dfselected['eventType'].iteritems():
    test= dfselected.loc[dfselected["eventType"]==values]
    #df6 = df6.append({'eventType': values}, ignore_index=True)
   # firstRound = False
    #k+= 1
 #   print("Len : ",name, len(test))   
    if (cnt<=3):
        lensum= lensum + len(dfselected.loc[dfselected["eventType"]==values])
        for i in range(0,len(test)+1):
        #for key, val in dfselected.loc[dfselected["eventType"]==values].iteritems():
            print("trace gran: ", cnt)
            k+= 1
            #exists = values in df6.eventType
        if(values not in df6['eventType'].values):
            df5 = df5.append({'caseID': "CID0" + str(j)}, ignore_index=True)
            df6 = df6.append({'eventType': values}, ignore_index=True)       
        else:
            continue
    else:
        cnt = 0
        j+= 1
#print("event list:", eventList)
    cnt+= 1
    df1.append(test)
    dfselected.drop(dfselected.loc[dfselected['eventType'] == values].index, inplace = True)    
print("No of iterations: ", k)
print("Sum of total length: ", lensum)
#dfc1 = pd.DataFrame(caseIDList, columns=['caseID'])
#dfc2 = pd.DataFrame(columns=['eventTLookUp'])
#dfc2 = pd.DataFrame(eventList, columns=['eventTLookUp'])
#dfc2= pd.Series(eventList, name='eventType')
df1.to_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_5_rrc04_V3.csv', index=False)

frames =[df5, df6]
df3 = pd.concat(frames, axis=1)
#print(dfc2)
#print(df.eventType.value_counts())

df3.to_csv('C:/2020-2021/Data/Datasets/Malaysian/caseIDLookUp.csv', index=False)
#caseID = df3["caseID"]
dfr = pd.merge(df3, df1, how="inner", on="eventType", validate="one_to_many")
dfro= dfr.sort_values(by=['timestamp'])
dfro.to_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_5_rrc04_MergedO.csv', index=False)

#df.head(25)


# In[ ]:


lst =[]
lst1 = []

df5 = pd.read_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_5_rrc04_V3.csv')
print("Length of df5: ", len(df5.eventType))
lst1 = df5["eventType"].value_counts()
df4 = pd.Series(lst, name='value')
df4 = pd.read_csv('C:/2020-2021/Data/Datasets/Malaysian/caseIDLookup.csv')
df5.eventType.isnull()


# In[ ]:


df5= df[df["eventType"]=="Evt021"]
df5


# In[ ]:





# In[ ]:





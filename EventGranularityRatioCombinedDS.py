#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
df = pd.read_csv('C:/2020-2021/Data/Datasets/CombinedDataset/dataset_combined_1_rrc04_EqualF.csv', delimiter=',')
df1 =pd.DataFrame(columns=["ann__egr", "nlri_egr", "withdrawals_egr", "origin_changes_egr", "number_rare_ases_egr", "as_path_max_egr", "imp_wd_egr", "rare_ases_max_egr",
                        "as_path_avg_egr", "nadas_egr", "flaps_egr", "dups_egr", "news_egr"])
announcements = df["announcements"].nunique()
nlri_ann = df["nlri_ann"].nunique()
withdrawals = df["withdrawals"].nunique()
origin_changes= df["origin_changes"].nunique()
number_rare_ases=df["number_rare_ases"].nunique()
as_path_max = df["as_path_max"].nunique()
imp_wd=df["imp_wd"].nunique()
rare_ases_max= df["rare_ases_max"].nunique()
as_path_avg =df["as_path_avg"].nunique()
nadas = df["nadas"].nunique()
flaps = df["flaps"].nunique()
dups = df["dups"].nunique()
news = df["news"].nunique()
ann_eventgr= 1- announcements/len(df)
nlri_eventgr = 1- nlri_ann/len(df)
withdrawals_eventgr = 1- withdrawals/len(df)
origin_changes_eventgr = 1 - origin_changes/len(df)
number_rare_ases_eventgr = 1 - number_rare_ases/len(df)
as_path_max_eventgr = 1 - as_path_max/len(df)
imp_wd_eventgr = 1 - imp_wd/len(df)
rare_ases_max_eventgr = 1- rare_ases_max/len(df)
as_path_avg_eventgr = 1 - as_path_avg/len(df)
nadas_eventgr = 1 - nadas/len(df)
flaps_eventgr = 1- flaps/len(df)
dups_eventgr = 1- dups/len(df)
news_eventgr = 1- news/len(df)

dict1 ={'ann__egr': ann_eventgr,'nlri_egr': nlri_eventgr,'withdrawals_egr': withdrawals_eventgr,
      'origin_changes_egr': origin_changes_eventgr,'number_rare_ases_egr': number_rare_ases_eventgr,'as_path_max_egr': as_path_max_eventgr,
     'imp_wd_egr':imp_wd_eventgr,'rare_ases_max_egr': rare_ases_max_eventgr,'as_path_avg_egr': as_path_avg_eventgr
       ,'nadas_egr': nadas_eventgr,'flaps_egr': flaps_eventgr,'dups_egr': dups_eventgr,'news_egr':news_eventgr}
dict2= {'announcements': announcements,'nlri_ann': nlri_ann,'withdrawals': withdrawals,
      'origin_changes': origin_changes,'number_rare_ases': number_rare_ases,'as_path_max': as_path_max,
     'imp_wd':imp_wd,'rare_ases_max': rare_ases_max,'as_path_avg': as_path_avg
       ,'nadas': nadas,'flaps': flaps,'dups': dups,'news':news}

df2= pd.DataFrame(dict1,index=[0])
df3 = pd.DataFrame(dict2,index=[0])
df4.append = pd.concat([df2,df3])

"""df2["ann__egr"] = df1.assign(ann_egr=ann_eventgr)
df2["nlri_egr"] = nlri_eventgr
df2["withdrawals_egr"] = withdrawals_eventgr
df2["origin_changes_egr"] = origin_changes_eventgr
df2["number_rare_ases_egr"] = number_rare_ases_eventgr
df2["as_path_max_egr"] = as_path_max_eventgr
df2["imp_wd_egr"] = imp_wd_eventgr
df2["rare_ases_max_egr"] = rare_ases_max_eventgr
df2["as_path_avg_egr"] = as_path_avg_eventgr
df2["nadas_egr"] = nadas_eventgr
df2["flaps_egr"] = flaps_eventgr
df2["dups_egr"] = dups_eventgr
df2["news_egr"] = news_eventgr"""
df4.to_csv('C:/2020-2021/Data/Datasets/CombinedDataset/event_granularity_ratio.csv')

print("Event Granularity: ", ann_eventgr, nlri_eventgr,withdrawals_eventgr, origin_changes_eventgr, number_rare_ases_eventgr)
print("No. of unique events: ", announcements, nlri_ann, withdrawals,origin_changes, number_rare_ases, as_path_max, imp_wd, rare_ases_max, as_path_avg, 
     nadas, flaps, dups, news)


# In[ ]:





# In[ ]:





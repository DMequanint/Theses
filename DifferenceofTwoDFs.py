#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df1 = pd.read_csv('C:/2020-2021/Data/Datasets/AWSLeak/dataset_aws-leak_15547_1_rrc04_balanced.csv', delimiter=',', index_col=False)
df2 = pd.read_csv('C:/2020-2021/Data/Datasets/AWSLeak/dataset_aws-leak_15547_1_rrc04_balancedF.csv', delimiter=',', index_col=False) 
df1[~df1.apply(tuple,1).isin(df2.apply(tuple,1))]


# In[ ]:





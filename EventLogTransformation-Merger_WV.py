#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import time
import datetime
df4 = pd.read_csv('C:/2020-2021/Data/Datasets/AWSLeak/dataset_aws-leak_15547_1_rrc04_balancedF.csv', delimiter=',')
df3= pd.read_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_1_rrc04_balancedF.csv', delimiter=',')
df1= pd.read_csv('C:/2020-2021/Data/Datasets/Nimda/dataset_nimda_513_1_rrc04_balancedF.csv', delimiter=',')
df2= pd.read_csv('C:/2020-2021/Data/Datasets/Slammer/dataset_slammer_513_1_rrc04_balancedF.csv', delimiter=',')
frames = [df1, df2, df3, df4]
df5= pd.concat(frames)
df5.to_csv('C:/2020-2021/Data/Datasets/CombinedDataset/dataset_combined_1_rrc04_balancedF.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





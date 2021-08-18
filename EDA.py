#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sweetviz
import pandas as pd
train = pd.read_csv('/home/dessalegn/Datasets/dataset_aws-leak_15547_1_rrc04.csv')
test = pd.read_csv('/home/dessalegn/Datasets/features-aws-leak-rrc04-15547-1.csv')
train.drop('Unnamed: 0', axis=1, inplace=True)
test.drop('Unnamed: 0', axis=1, inplace=True)
my_report = sweetviz.compare([train, "Train"], [test, "Test"], "class")
my_report.show_html("Report.html")


# In[ ]:





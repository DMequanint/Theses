#!/usr/bin/env python
# coding: utf-8

# In[40]:


import sweetviz as sv
import pandas as pd
df = pd.read_csv('/home/dessalegn/Datasets/dataset_aws-leak_15547_1_rrc04.csv')
df.drop(["Unnamed: 0", 'timestamp'], axis=1, inplace=True)
#print (df.shape)
#cols = df.columns
#df1= pd.DataFrame(columns=cols)
#pd1 = df.iloc[0,1]
#df.diff().head(100)
#print(df.iloc[0])
df1 = pd.DataFrame([df.iloc[1]], columns=df.columns)
#print (df1)

cols = df.columns.difference(['class'])
df[cols]=df[cols].diff()

df2= df1.append(df.dropna())
df2['class'] =df2['class'].astype(int)
print(df2)

#df1['class'] = df1['class'].astype(int).abs()

#df1= df[df.columns[~df.columns.isin(['class'])]].diff()
sv.config_parser.read("Override.ini")
#df.iloc[0,]
my_report2= sv.analyze(df2, "class")
my_report2.show_html("RandomReport.html", open_browser=True, layout= 'vertical', scale=0.8)
#df.head()


# In[ ]:





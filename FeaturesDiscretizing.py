#!/usr/bin/env python
# coding: utf-8

# In[69]:


from pandas import read_csv
from pandas import DataFrame
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib import pyplot
# load dataset

df = pd.read_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04.csv', delimiter=',', index_col=False)
df_new = df[df.columns.difference(['class', 'timestamp', "Unnamed: 0"])]
cols =df_new.columns

dataset = df_new


# perform a k-means discretization transform of the dataset
#trans = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
trans = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')
datasett = trans.fit_transform(dataset)
# convert the array back to a dataframe

datasetfull = pd.DataFrame(datasett, columns=df_new.columns)



#print(cols)
datasetfull['class']= df['class']
datasetfull['timestamp'] = df['timestamp']
#datasetfull = pd.concat([datasett, df['class'], df['timestamp']], axis = 1)
datasetfull.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04_uniform.csv')


# In[ ]:





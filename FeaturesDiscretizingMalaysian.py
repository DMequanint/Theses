#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.discretisation import EqualFrequencyDiscretiser
# load dataset

#df = pd.read_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04.csv', delimiter=',', index_col=False)
df =  pd.read_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_1_rrc04_balancedF.csv', delimiter=',', index_col=False)
#df_new = df[df.columns.difference(['class', 'timestamp', "Unnamed: 0"])]
df_new = df.loc[:, ~df.columns.isin(['timestamp','Unnamed: 0'])]
cols =df_new.columns

dataset = df_new


# perform a k-means discretization transform of the dataset
#trans = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
#trans = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
trans = EqualFrequencyDiscretiser(q=100, variables=['announcements', 'class', 'nlri_ann', 'withdrawals', 'origin_changes', 'number_rare_ases', 'as_path_max', 'imp_wd',
                                                  'rare_ases_max', 'as_path_avg', 'nadas', 'flaps', 'dups', 'news'])
#trans = EqualWidthDiscretiser(bins=20, variables=['announcements', 'nlri_ann', 'withdrawals', 'origin_changes', 'number_rare_ases', 'as_path_max', 'imp_wd',
#                                                  'rare_ases_max', 'as_path_avg', 'nadas', 'flaps', 'dups', 'news'])
datasett = trans.fit_transform(dataset)
# convert the array back to a dataframe

datasetfull = pd.DataFrame(datasett, columns=df_new.columns)
#print(datasetfull["withdrawals"])

print(datasetfull["announcements"].value_counts(bins=100, sort=False),datasetfull["withdrawals"].value_counts(bins=100, sort=False), datasetfull["nlri_ann"].value_counts(bins=100, sort=False))
#print(cols)
datasetconca= pd.concat((datasetfull, df['timestamp']), axis=1)
#datasetfull['class']= df['class']
#datasetfull['timestamp'] = df['timestamp']
#datasetfull = pd.concat([datasett, df['class'], df['timestamp']], axis = 1)
datasetconca.to_csv('C:/2020-2021/Data/Datasets/Malaysian/dataset_malaysian-telecom_513_1_rrc04_EqualF.csv')


# In[ ]:





# In[ ]:





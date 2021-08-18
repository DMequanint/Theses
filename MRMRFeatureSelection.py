#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pymrmre import mrmr
from sklearn import preprocessing
from sklearn.decomposition import PCA
df = pd.read_csv('/home/dessalegn/Datasets/dataset_multi_aws-leak_15547_1_rrc04.csv')
df.drop(['Unnamed: 0', 'timestamp'], axis =1, inplace=True)
#X = df.drop(['class'], axis =1)
#y = pd.DataFrame(df['class'])
df1 = pd.DataFrame(df.drop(['class'], axis =1))
std_scale =preprocessing.StandardScaler().fit(df1)
df_std = std_scale.transform(df1)
dfstd = pd.DataFrame(df_std, columns=df1.columns)
X= dfstd
y = pd.DataFrame(df['class'])
#solutions = mrmr.mrmr_ensemble(features=X,targets=Y,fixed_features=['f1'],category_features=['f4','f5'],solution_length=5,solution_count=3)
solutions = mrmr.mrmr_ensemble(features=X,targets=y,fixed_features=['announcements'],category_features=[],solution_length=10,solution_count=3)
for i in range(3):
    print(solutions.iloc[0][i])


# In[29]:


import pandas as pd
from mrmr import mrmr_classif 
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('/home/dessalegn/Datasets/dataset_multi_aws-leak_15547_1_rrc04.csv', index_col=None)
X,y= make_classification(n_samples=1000, n_features=46, n_informative=10, n_redundant = 36)
X = df.drop(['Unnamed: 0', 'class', 'timestamp2'], axis =1)
y = df['class']
selected_features = mrmr_classif(X,y, K = 15)
#selected_features.plot.barh("Test", figsize=(8,8))
plt.xlabel('Feature Importance Score')
plt.show()
print(df.shape)
print(selected_features)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[44]:


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import pandas as pd
# define dataset
# for reproducibility purposes
seed=100
# SMOTE number of neighbors
k=5
df = pd.read_csv('C:/2020-2021/Data/Nimda/nimda-rrc00.csv', delimiter=',')
X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.8, 0.2],
    n_informative=12, n_redundant=1, flip_y=0,
    n_features=49, n_clusters_per_class=1,
    n_samples=3462, random_state=10
)
"""X,y= make_classification(n_samples=2770, n_features=49, n_redundant=0,
   n_informative=12, n_clusters_per_class=1,
  class_sep=1.0, flip_y=0.06, random_state=100)"""
X=df.drop(['timestamp'], 1)
# summarize class distribution
print(Counter(y))
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')

# fit and apply the transform
#X_over, y_over = oversample.fit_resample(X, y)
sm = SMOTE(sampling_strategy='minority', k_neighbors=k, random_state=seed)
X_res, y_res = sm.fit_resample(X, y)
df2 = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
df2.columns = df.columns
df2.to_csv('C:/2020-2021/Data/Nimda/balnaced_base.csv', index=False, encoding='utf-8')
# summarize class distribution
print(Counter(y_over))
count1 = df[df['class']==1].count()['class'] 
count2 = df[df['class']==0].count()['class']
count3 = df2[df2['class']==1].count()['class']
count4 = df2[df2['class']==0].count()['class']
print("Imbalanced: Anomaly: {0}, regular: {1}".format(count1,count2) )
print("Balanced: Anomaly: {0}, regular: {1}".format(count3,count4) )
#df2.head(2000)


# In[ ]:





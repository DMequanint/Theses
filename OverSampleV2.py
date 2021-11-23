#!/usr/bin/env python
# coding: utf-8

# In[5]:


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
# define dataset
# for reproducibility purposes
seed=100
# SMOTE number of neighbors
k=5
df = pd.read_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_5_rrc04.csv', delimiter=',')
X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.7, 0.3],
    n_informative=15, n_redundant=1, flip_y=0,
    n_features=50, n_clusters_per_class=1,
    n_samples=10000, random_state=10
)
""" X,y= make_classification(n_samples=10000, n_features=50, n_redundant=0,
   n_informative=15, n_clusters_per_class=1,
  class_sep=1.0, flip_y=0.06, weights=[0.7, 0.3],random_state=100)
df.drop("Unnamed: 0",axis=1, inplace=True)"""
#X=df.drop(['timestamp'], 1)
# summarize class distribution
#print(Counter(y))
# define oversampling strategy
#oversample = RandomOverSampler(sampling_strategy='minority')
#X_over, y_over = oversample.fit_resample(X, y)
# fit and apply the transform
#X_over, y_over = oversample.fit_resample(X, y)
sm = SMOTE(sampling_strategy='minority', k_neighbors=k, random_state=seed)
X_res, y_res = sm.fit_resample(X, y)
#X_resa, y_resa = ADASYN().fit_resample(X, y)
df2 = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
#df3 = pd.concat([pd.DataFrame(X_resa), pd.DataFrame(y_resa)], axis=1)
#df4 = pd.concat([pd.DataFrame(X_over), pd.DataFrame(y_over)], axis=1)
df2.columns = df.columns
#df3.columns = df.columns
#df4.columns = df.columns
df2.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_5_rrc04_balanced.csv', index=False, encoding='utf-8')
#df3.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_5_rrc04_balanceda.csv', index=False, encoding='utf-8')
#df4.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_5_rrc04_balancedo.csv', index=False, encoding='utf-8')
# summarize class distribution
#print(Counter(y_over))
count1 = df[df['class']==1].count()['class'] 
count2 = df[df['class']==0].count()['class']
count3 = df2[df2['class']==1].count()['class']
count4 = df2[df2['class']==0].count()['class']
print("Imbalanced: Anomaly: {0}, regular: {1}".format(count1,count2) )
print("Balanced: Anomaly: {0}, regular: {1}".format(count3,count4) )
#df2.head(2000)


# In[ ]:





# In[ ]:





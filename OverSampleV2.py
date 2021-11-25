#!/usr/bin/env python
# coding: utf-8

# In[24]:


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from sklearn import preprocessing
import pandas as pd
from numpy import where
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
# define dataset
# for reproducibility purposes
seed=100
# SMOTE number of neighbors
k=5
df = pd.read_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04.csv', delimiter=',')
#df= pd.read_csv('C:/2020-2021/Data/Nimda/nimda-rrc00.csv', delimiter=',') 

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.7, 0.3],
    n_informative=15, n_redundant=1, flip_y=0,
    n_features=49, n_clusters_per_class=1,
    n_samples=10000, random_state=10
)
""" X,y= make_classification(n_samples=10000, n_features=50, n_redundant=0,
   n_informative=15, n_clusters_per_class=1,
  class_sep=1.0, flip_y=0.06, weights=[0.7, 0.3],random_state=100)"""

df.drop(["Unnamed: 0", "timestamp"],axis=1, inplace=True)
#df['timestamp']=pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
X= df.loc[:, df.columns !='class']
y= df['class']

#X=df.drop(['timestamp'], 1)
# summarize class distribution

print(Counter(y))
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
sm = SMOTE(sampling_strategy='minority', k_neighbors=k, random_state=seed)
#sm =BorderlineSMOTE()
X_res, y_res = sm.fit_resample(X, y)
X_resa, y_resa = ADASYN().fit_resample(X, y)

df2 = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
df3 = pd.concat([pd.DataFrame(X_resa), pd.DataFrame(y_resa)], axis=1)
df4 = pd.concat([pd.DataFrame(X_over), pd.DataFrame(y_over)], axis=1)
#df2.columns = df.columns
#df3.columns = df.columns
#df4.columns = df.columns
dfstdd= pd.DataFrame(df2.loc[:, df2.columns!='class'])
std_scale =preprocessing.StandardScaler().fit(dfstdd)
df_std = std_scale.transform(dfstdd)
dfstd = pd.DataFrame(df_std, columns=dfstdd.columns)

dfstd2= pd.concat((dfstd, df2['class']), axis=1)
dfstd2.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04_standardized.csv', index=False, encoding='utf-8')
df2.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04_balanced.csv', index=False, encoding='utf-8')
df3.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04_balanceda.csv', index=False, encoding='utf-8')
df4.to_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04_balancedo.csv', index=False, encoding='utf-8')
# summarize class distribution
print(Counter(y_res))
#print(Counter(y_resa))
print(Counter(y_over))
count1 = df[df['class']==1].count()['class'] 
count2 = df[df['class']==0].count()['class']
count3 = df2[df2['class']==1].count()['class']
count4 = df2[df2['class']==0].count()['class']
#count5 = df2[df3['class']==1].count()['class']
#count6 = df2[df3['class']==0].count()['class']
#count7 = df2[df4['class']==1].count()['class']
#count8 = df2[df4['class']==0].count()['class']

print("Imbalanced: Anomaly: {0}, regular: {1}".format(count1,count2) )
print("Balanced: Anomaly: {0}, regular: {1}".format(count3,count4) )
#print("Balanced: Anomaly: {0}, regular: {1}".format(count5,count6) )
#print("Balanced: Anomaly: {0}, regular: {1}".format(count7,count8) )
#df2.head(2000)
counter = Counter(y_res)
for label, _ in counter.items():
	row_ix = where(y_res == label)[0]
	pyplot.scatter(X_res.iloc[row_ix, 0], X_res.iloc[row_ix, 1], label=str(label))
pyplot.savefig('C:/2020-2021/Data/Datasets/balanced.png')
"""yy=dfstd2['class']
xx=dfstd2.loc[:, dfstd2.columns!='class']
counter = Counter(yy)
for label, _ in counter.items():
	row_ix = where(yy == label)[0]
	pyplot.scatter(xx.iloc[row_ix, 0], xx.iloc[row_ix, 1], label=str(label))
pyplot.savefig('C:/2020-2021/Data/Datasets/standardizedbalanced.png')"""


# In[ ]:





# In[ ]:





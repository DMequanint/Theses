#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
k=3
df = pd.read_csv('C:/2020-2021/Data/Datasets/dataset_aws-leak_15547_1_rrc04.csv', delimiter=',')
#df= pd.read_csv('C:/2020-2021/Data/Nimda/nimda-rrc00.csv', delimiter=',') 
"""
X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.7, 0.3],
    n_informative=15, n_redundant=1, flip_y=0,
    n_features=49, n_clusters_per_class=1,
    n_samples=10000, random_state=10
)
 X,y= make_classification(n_samples=10000, n_features=50, n_redundant=0,
   n_informative=15, n_clusters_per_class=1,
  class_sep=1.0, flip_y=0.06, weights=[0.7, 0.3],random_state=100)"""

df.drop("Unnamed: 0",axis=1, inplace=True)
#df['timestamp']=pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
#X= df.loc[:, df.columns !='class']
X = df.loc[:, ~df.columns.isin(['class', 'timestamp'])]
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

df2 = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res), df['timestamp']], axis=1)
df3 = pd.concat([pd.DataFrame(X_resa), pd.DataFrame(y_resa)], axis=1)
df4 = pd.concat([pd.DataFrame(X_over), pd.DataFrame(y_over)], axis=1)
#df2.columns = df.columns
#df3.columns = df.columns
#df4.columns = df.columns
dfscaled = pd.DataFrame(df2.loc[:, ~df2.columns.isin(['class', 'timestamp'])])
std_scale = preprocessing.StandardScaler().fit(dfscaled)
df_std = std_scale.transform(dfscaled)
dfstd = pd.DataFrame(df_std, columns=dfscaled.columns)
dfstd2= pd.concat((dfstd, df2['class'], df2['timestamp']), axis=1)
minmax_scale = preprocessing.MinMaxScaler().fit(dfscaled)
df_minmax = minmax_scale.transform(dfscaled)
dfmmin = pd.DataFrame(df_minmax, columns=dfscaled.columns)
dfmmin2 = pd.concat((dfmmin, df2['class'],df2['timestamp']), axis=1)
dfstd2.to_csv('C:/2020-2021/Data/Datasets/AWSLeakPlot/dataset_aws-leak_15547_1_rrc04_standardized.csv', index=False, encoding='utf-8')
dfmmin2.to_csv('C:/2020-2021/Data/Datasets/AWSLeakPlot/dataset_aws-leak_15547_1_rrc04_maxmin.csv', index=False, encoding='utf-8')
df2.to_csv('C:/2020-2021/Data/Datasets/AWSLeakPlot/dataset_aws-leak_15547_1_rrc04_balanced.csv', index=False, encoding='utf-8')
df3.to_csv('C:/2020-2021/Data/Datasets/AWSLeakPlot/dataset_aws-leak_15547_1_rrc04_balanceda.csv', index=False, encoding='utf-8')
df4.to_csv('C:/2020-2021/Data/Datasets/AWSLeakPlot/dataset_aws-leak_15547_1_rrc04_balancedo.csv', index=False, encoding='utf-8')
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
"""
counter = Counter(y)
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	#print(X_over.iloc[row_ix, 2],X_over.iloc[row_ix, 3],X_over.iloc[row_ix, 4],X_over.iloc[row_ix, 5],X_over.iloc[row_ix, 30],X_over.iloc[row_ix, 31], X_over.iloc[row_ix, 34],X_over.iloc[row_ix, 35],X_over.iloc[row_ix, 36],X_over.iloc[row_ix, 37],X_over.iloc[row_ix, 41],X_over.iloc[row_ix, 48])
	#pyplot.scatter([X.iloc[row_ix, 2],X.iloc[row_ix, 3],X.iloc[row_ix, 4],X.iloc[row_ix, 5],X.iloc[row_ix, 30],X.iloc[row_ix, 31]], [X.iloc[row_ix, 34],X.iloc[row_ix, 35],X.iloc[row_ix, 36],X.iloc[row_ix, 37],X.iloc[row_ix, 41],X.iloc[row_ix, 48]], label=str(label))
	pyplot.scatter(X.iloc[row_ix,2],X.iloc[row_ix, 36], label=str(label))
	pyplot.legend()
	pyplot.savefig('C:/2020-2021/Data/Datasets/AWSLeakPlot/Imbalanced.png')

"""
groups = df.groupby("class")
for name, group in groups:
    #pyplot.scatter([group["announcements"], group["withdrawals"], group["origin_changes"],group["nlri_ann"]], [group["as_path_max"],group["number_rare_ases"], group["as_path_avg"],group["rare_ases_max"]], label=name)
    #pyplot.plot(group["announcements","as_path_avg", "as_path_max", "dups", "flaps", "imp_wd"], group["nadas", "news", "nlri_ann", "number_rare_ases", "origin_changes", "withdrawals"], marker="o", linestyle="", label=name)
    pyplot.scatter(group["announcements"], group["nlri_ann"], label=name)
    pyplot.savefig('C:/2020-2021/Data/Datasets/AWSLeakPlot/Imbalancednew.png')                                                                                                    
pyplot.legend()


"""
counter = Counter(y_res)
for label, _ in counter.items():
	row_ix = where(y_res == label)[0]
	print(row_ix)
	#pyplot.scatter([X_res.iloc[row_ix, 2],X_res.iloc[row_ix, 3],X_res.iloc[row_ix, 4],X_res.iloc[row_ix, 5],X_res.iloc[row_ix, 30],X_res.iloc[row_ix, 31]], [X_res.iloc[row_ix, 34],X_res.iloc[row_ix, 35],X_res.iloc[row_ix, 36],X_res.iloc[row_ix, 37],X_res.iloc[row_ix, 41],X_res.iloc[row_ix, 48]], s=60, label=str(label))
	pyplot.scatter(X_res.iloc[row_ix,2], X_res.iloc[row_ix, 36], label=str(label))
	pyplot.legend()
	pyplot.savefig('C:/2020-2021/Data/Datasets/AWSLeakPlot/SMOTEbalanced.png')


ys=dfstd2['class']
Xs=dfstd2.loc[:, dfstd2.columns!='class']
counter = Counter(ys)
for label, _ in counter.items():
	row_ix = where(ys == label)[0]
	#pyplot.scatter([Xs.iloc[row_ix, 2],Xs.iloc[row_ix, 3],Xs.iloc[row_ix, 4],Xs.iloc[row_ix, 5],Xs.iloc[row_ix, 30],Xs.iloc[row_ix, 31]], [Xs.iloc[row_ix, 34],Xs.iloc[row_ix, 35],Xs.iloc[row_ix, 36],Xs.iloc[row_ix, 37],Xs.iloc[row_ix, 41],Xs.iloc[row_ix, 48]], label=str(label))
	pyplot.scatter(Xs.iloc[row_ix,2],Xs.iloc[row_ix, 36], label=str(label))
	pyplot.legend()
	pyplot.savefig('C:/2020-2021/Data/Datasets/AWSLeakPlot/standardizedbalanced.png')

ym=dfmmin2['class']
Xm=dfmmin2.loc[:, dfmmin2.columns!='class']
counter = Counter(ym)
for label, _ in counter.items():
	row_ix = where(ym == label)[0]
	#pyplot.scatter([Xm.iloc[row_ix, 2],Xm.iloc[row_ix, 3],Xm.iloc[row_ix, 4],Xm.iloc[row_ix, 5],Xm.iloc[row_ix, 30],Xm.iloc[row_ix, 31]], [Xm.iloc[row_ix, 34],Xm.iloc[row_ix, 35],Xm.iloc[row_ix, 36],Xm.iloc[row_ix, 37],Xm.iloc[row_ix, 41],Xm.iloc[row_ix, 48]], label=str(label))
	pyplot.scatter(Xm.iloc[row_ix,2],Xm.iloc[row_ix, 36], label=str(label))
	pyplot.legend()
	pyplot.savefig('C:/2020-2021/Data/Datasets/AWSLeakPlot/minmaxbalanced.png')

  

counter = Counter(y_resa)
for label, _ in counter.items():
	row_ix = where(y_resa == label)[0]
	#pyplot.scatter([X_resa.iloc[row_ix, 2],X_resa.iloc[row_ix, 3],X_resa.iloc[row_ix, 4],X_resa.iloc[row_ix, 5],X_resa.iloc[row_ix, 30],X_resa.iloc[row_ix, 31]], [X_resa.iloc[row_ix, 34],X_resa.iloc[row_ix, 35],X_resa.iloc[row_ix, 36],X_resa.iloc[row_ix, 37],X_resa.iloc[row_ix, 41],X_resa.iloc[row_ix, 48]], label=str(label))
	pyplot.scatter(X_resa.iloc[row_ix,2],X_resa.iloc[row_ix, 36], label=str(label))
	pyplot.legend()
	pyplot.savefig('C:/2020-2021/Data/Datasets/AWSLeakPlot/adasynbalanced.png')


counter = Counter(y_over)
for label, _ in counter.items():
	row_ix = where(y_over == label)[0]
	#print(X_over.iloc[row_ix, 2],X_over.iloc[row_ix, 3],X_over.iloc[row_ix, 4],X_over.iloc[row_ix, 5],X_over.iloc[row_ix, 30],X_over.iloc[row_ix, 31], X_over.iloc[row_ix, 34],X_over.iloc[row_ix, 35],X_over.iloc[row_ix, 36],X_over.iloc[row_ix, 37],X_over.iloc[row_ix, 41],X_over.iloc[row_ix, 48])
	#pyplot.scatter([X_over.iloc[row_ix, 2],X_over.iloc[row_ix, 3],X_over.iloc[row_ix, 4],X_over.iloc[row_ix, 5],X_over.iloc[row_ix, 30],X_over.iloc[row_ix, 31]], [X_over.iloc[row_ix, 34],X_over.iloc[row_ix, 35],X_over.iloc[row_ix, 36],X_over.iloc[row_ix, 37],X_over.iloc[row_ix, 41],X_over.iloc[row_ix, 48]], label=str(label))
	#pyplot.scatter([X_over.iloc[row_ix, 2], X_over.iloc[row_ix, 41]],[X_over.iloc[row_ix, 36],X_over.iloc[row_ix, 48]], label=str(label))
	pyplot.scatter(X_over.iloc[row_ix,2],X_over.iloc[row_ix, 36], label=str(label))
	pyplot.legend()
	pyplot.savefig('C:/2020-2021/Data/Datasets/AWSLeakPlot/overbalanced.png') 
"""

pyplot.show()


# In[ ]:





# In[ ]:





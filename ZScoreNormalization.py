#!/usr/bin/env python
# coding: utf-8

# In[28]:


import os.path
from os import path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from  matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

"""df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
    )
df1= df.describe()
df2 = df1.drop(["count","mean", "25%", "50%", "75%"], axis=0)
df2.drop("Unnamed: 0", axis=1, inplace=True)
df2.to_csv('C:/2020-2021/Data/Nimda/describe.csv') """
df = pd.read_csv('C:/2020-2021/Data/NimdaMain/dataset_nimda_513_1_rrc04.csv', delimiter=',')
df.drop(['class','timestamp',"Unnamed: 0"], axis=1, inplace=True)
#df.drop('timestamp', axis=1, inplace=True)
#std_scale =preprocessing.StandardScaler().fit(df[['announcements', 'rare_ases_avg']])
std_scale =preprocessing.StandardScaler().fit(df)
#y= df['class']
#df_std = std_scale.transform(df[['announcements','rare_ases_avg']])
df_std = std_scale.transform(df)

#minmax_scale = preprocessing.MinMaxScaler().fit(df[['announcements','rare_ases_avg']])
minmax_scale = preprocessing.MinMaxScaler().fit(df)
#df_minmax = minmax_scale.transform(df[['announcements','rare_ases_avg']])
df_minmax = minmax_scale.transform(df)
print (df_std.shape)
""""def plot():
    plt.figure(figsize=(8,6))

    #plt.scatter(df['announcements'], df['rare_ases_avg'],
            #color='green', label='input scale', alpha=0.5)
    plt.scatter(df[:,1], df[:,0:50],
            color='green', label='input scale', alpha=0.5)

    #plt.scatter(df_std[:,0], df_std[:,1], color='red',
        #    label='Standardized', alpha=0.3)
    plt.scatter(df_std[:,0], df_std[:,50], color='red',
            label='Standardized', alpha=0.3)
    plt.title('BGP Nimda Anomaly dataset')
    plt.xlabel('Announcements')
    plt.ylabel('Rare ASes Average')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()
plot()
plt.show()"""

#df.plot.barh(x="announcements", y=df.columns, figsize=(20, 20))
#x=0
#for col in df0.columns:
#pd.plotting(df, kind='barh')
    #df[col].plot(kind="barh")
 #   x=x+1
  #  if(x==10):
   #     break"""
#df.plot(kind='hist')
#plt.legend(loc="lower left",bbox_to_anchor=(0.8,1.0))
#plt.figure(figsize=(8,6))
#plt.show()
#plt.show()
#df.plot(kind='bar')
#plt.hist(df, bins=100)
#plt.show()


dfstd = pd.DataFrame(df_std, columns=df.columns)
dfstd.to_csv('C:/2020-2021/Data/NimdaMain/DescriptiveStat/standardized.csv')
dfstdS = dfstd.describe() 
dfstdS.drop(["count", "25%", "50%", "75%"], axis=0, inplace=True)
dfstdS.describe()
df1 = df.describe()
df1.drop(["count", "25%", "50%", "75%"], axis=0, inplace=True)

if (path.exists('C:/2020-2021/Data/NimdaMain/DescriptiveStat/beforescaling.csv')):
    print("File Already exists")
    print
else:
    df1.iloc[:, :9].to_csv('C:/2020-2021/Data/NimdaMain/DescriptiveStat/beforescaling.csv')
    
#df1.plot(linewidth=1)   
mean,std,min,max = [],[],[],[]
for i in range(len(df1.columns)):
    mean.append(df1.iloc[0, i])
    std.append(df1.iloc[1, i])
    min.append(df1.iloc[2, i])
    max.append(df1.iloc[3, i])



#print(mean)



df3 = pd.DataFrame(mean, columns=["Mean"])
df3["Std"] = pd.DataFrame(std)
df3["Min"] = pd.DataFrame(min)
df3["Max"] = pd.DataFrame(max)
df3.insert(0, 'col', df.columns)
df3.iloc[0:,:].to_csv('C:/2020-2021/Data/NimdaMain/DescriptiveStat/beforescalingfile.csv')
#plt.legend(loc='upper right')
means,stds,mins,maxs = [],[],[],[]
for i in range(len(dfstdS.columns)):
    means.append(dfstdS.iloc[0, i])
    stds.append(dfstdS.iloc[1, i])
    mins.append(dfstdS.iloc[2, i])
    maxs.append(dfstdS.iloc[3, i])
df4 = pd.DataFrame(means, columns=["Mean"])
df4["Std"] = pd.DataFrame(stds)
df4["Min"] = pd.DataFrame(mins)
df4["Max"] = pd.DataFrame(maxs)
df4.insert(0, 'col', df.columns)
df4.iloc[0:,:].to_csv('C:/2020-2021/Data/NimdaMain/DescriptiveStat/afterscalingfile.csv')
#df3.plot(linewidth=1)
#df3.plot(y=["Mean","Std","Min","Max"],subplots=True, layout=(2,2), sharey=True,style=['-', '--', '-.', ':'])
#df4.plot(linewidth=1)
#df4.plot(y=["Mean","Std","Min","Max"],subplots=True, layout=(2,2), sharey=True,style=['-', '--', '-.', ':'])


N = 49
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2
#plt.scatter(df3.Mean, df4.Mean, s=area, c=colors, alpha=0.5)
size = [100,500,100,500]
#plt.scatter(x=df["announcements"], y=df["withdrawals"],s=size, c='lightblue')
#plt.scatter(x=dfstd["announcements"], y=dfstd["withdrawals"],s=size, c='lightblue')


plt.title('Features Mean Before and After Z-score Normalization')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = dfstd["announcements"], y = dfstd["withdrawals"],color = "blue", edgecolors = "white", linewidths = 0.1, alpha = 0.7)
plt.xlabel('Features Mean Before Z-Score Normalization')
plt.ylabel('Features Mean After Z-Score Normalization')

plt.savefig('C:/2020-2021/Data/NimdaMain/DescriptiveStat/mean.png')
plt.show()
df3.head(50)
#plt.tight_layout()


#df4.head()
#df3 = df1.drop(["count", "25%", "50%", "75%"], axis=0)
#df3.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





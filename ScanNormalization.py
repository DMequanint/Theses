#!/usr/bin/env python
# coding: utf-8

# In[12]:


# D'Agostino and Pearson's Test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import normaltest
import pandas as pd
# seed the random number generator
seed(1)
# generate univariate observations
#data = 5 * randn(100) + 50
df = pd.read_csv('C:/2020-2021/Data/Nimda/nimda-rrc00.csv', delimiter=',')
data = df.drop('timestamp', 1)
# normality test
stat, p = normaltest(data)
#print('Statistics=%.3f, p=%.3f' % (stat, p))
print ("P value is ", p)
# interpret
alpha = 0.05
if p.any() > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
    
else:
	print('Sample does not look Gaussian (reject H0)')


# In[ ]:





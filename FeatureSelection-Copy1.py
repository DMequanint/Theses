#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from ggplot import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
#df = pd.read_csv('C:/2020-2021/Data/dataset_as9121_12793_1_rrc05.csv', delimiter=',')
df = pd.read_csv('C:/2020-2021/Data/Nimda/nimda-rrc00.csv', delimiter=',')
#mydateparser = lambda x: pd.datetime.strptime(x, "%Y %m %d %H:%M:%S")
#df = pd.read_csv("C:/2020-2021/Data/features-nimda-rrc04-513-1.csv", sep='\t', names=['timestamp'], parse_dates=['timestamp'], date_parser=mydateparser)
#print(df.describe(include = 'all'))
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#import matplotlib as mpl
#import seaborn as sns
#from sklearn.feature_selection import SelectFromModel
#df = pd.read_csv('C:/2020-2021/RelevantThesisArticles/SelectedArticles/Shared/BGP/Data/dataset_as9121_12793_1_rrc05.csv', delimiter=",")
#X_train,y_train,X_test,y_test = train_test_split(df,test_size=0.3)
#df.info()
#X= df[:, 6:50]

#df.info()
y= df['class'].values
df.drop("Unnamed: 0",axis=1, inplace=True)
print(y)
#df['timestamp'] = pd.to_datetime(df['timestamp']), format='%Y %m %d %H:%M:%S')
#df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
#df['timestamp'] = df['timestamp'].astype('float')
X= df.drop(['class', 'timestamp'], 1)
print(X.shape)
feat_labels = X.columns
#print (str(df['timestamp']))

#cols = list(df.columns)
#df = df[cols[0:6] + cols[7:52]]
#X= df[cols[0:6] + cols[7:52]]
#df1=X
#df1.head(5)
#X_train = df[cols[0,52]]
#print(X_train)
#y_train = my_training_data[:
#df1.info()
#df[5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
#X_train,y_train,X_test,y_test = train_test_split(X, y, test_size=0.3)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#clf = SelectFromModel(RandomForestClassifier(n_estimators=100))
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf = clf.fit(X_train, y_train)
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)
#plt.barh(feat_labels, clf.feature_importances_)
sfm = SelectFromModel(clf, threshold=0.015)
sfm.fit(X_train, y_train)
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])

df_feature_importance = pd.DataFrame(clf.feature_importances_, index=feat_labels, columns=['feature importance']).sort_values('feature importance', ascending=False)
print(df_feature_importance)
df_feature_importance.plot(kind='bar');

#clf.get_support()
#selected_features= X_train.columns[(clf.get_support())]
#print(len(selected_features))
#print(selected_features)
print("RF train accuracy: %0.3f" %clf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % clf.score(X_test, y_test))
#print(f"Numbers of train instances by class: {np.bincount(y_train)}")
#print(f"Numbers of test instances by class: {np.bincount(y_test)}")
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[2]:


#import pandas as pd
#X_train = pd.DataFrame(X)
#y_train = pd.DataFrame(y)

from xgboost              import XGBClassifier
from sklearn.ensemble     import ExtraTreesClassifier
from sklearn.tree         import ExtraTreeClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.ensemble     import BaggingClassifier
from sklearn.ensemble     import AdaBoostClassifier
from sklearn.ensemble     import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from xgboost.core   import XGBoostError
import numpy  as np
#from lightgbm             import LGBMClassifier

def plot_feature_importances(clf, X_train, y_train=None, 
                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):
    '''
    plot feature importances of a tree-based sklearn estimator
    
    Note: X_train and y_train are pandas DataFrames
    
    Note: Scikit-plot is a lovely package but I sometimes have issues
              1. flexibility/extendibility
              2. complicated models/datasets
          But for many situations Scikit-plot is the way to go
          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html
    
    Parameters
    ----------
        clf         (sklearn estimator) if not fitted, this routine will fit it
        
        X_train     (pandas DataFrame)
        
        y_train     (pandas DataFrame)  optional
                                        required only if clf has not already been fitted 
        
        top_n       (int)               Plot the top_n most-important features
                                        Default: 10
                                        
        figsize     ((int,int))         The physical size of the plot
                                        Default: (8,8)
        
        print_table (boolean)           If True, print out the table of feature importances
                                        Default: False
        
    Returns
    -------
        the pandas dataframe with the features and their importance
        
    Author
    ------
        George Fisher
    '''
    
    __name__ = "plot_feature_importances"
   
    try: 
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train, y_train)

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                    format(clf.__class__.__name__))
                
    except (XGBoostError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())
            
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp

clfs = [XGBClassifier(),              
        ExtraTreesClassifier(),       ExtraTreeClassifier(),
        BaggingClassifier(),          DecisionTreeClassifier(),
        GradientBoostingClassifier(), 
        AdaBoostClassifier(),         RandomForestClassifier()]

for clf in clfs:
    try:
        _ = plot_feature_importances(clf, X_train, y_train, top_n=X_train.shape[1], title=clf.__class__.__name__)
    except AttributeError as e:
        print(e)
#df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in clf.estimators_], columns=feat_labels)
#df_feature_all.head()


# In[ ]:





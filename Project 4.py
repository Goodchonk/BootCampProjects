#!/usr/bin/env python
# coding: utf-8

# # Project 4 Bank Survey

# In[2]:


import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from sklearn import feature_selection as fs
from sklearn import metrics
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math

get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


dsf.dtypes


# In[3]:


dsf = pd.read_csv('bank.csv')
dsf.rename(index=str, columns={'y':'subbed'}, inplace = True)
print(dsf.shape)
dsf.head(20)


# In[4]:


limit = dsf.poutcome == 'other'
data = dsf.drop(dsf[limit].index, axis = 0, inplace = False)
data[['job', 'education']] =  data[['job', 'education']].replace(['unkown'], 'other')


# In[24]:



dsf.describe()


# ## Balance and Age

# In[6]:


def hist_plot(dsf, cols, col_x = 'age'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.histplot(x=col_x, y=col, data=dsf)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
        
num_cols = ['balance']
hist_plot(dsf, num_cols)


# In[7]:


sns.histplot(dsf.age, bins = 20)


# Note: that the spread of age and balances, with the majority appearing to be from 30 to 40yrs of age.

# In[8]:


def plot_box(dsf, cols, col_x = 'subbed'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(x=col_x, y=col, data=dsf)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
        
num_cols = ['age', 'balance', 'duration', 'campaign', 'previous']
plot_box(dsf, num_cols)


# In[9]:


def plot_violin(dsf, cols, col_x = 'subbed'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(x=col_x, y=col, data=dsf)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
    
plot_violin(dsf, num_cols)


# In[10]:


import numpy as np
cat_cols = ['job','marital','education','default','housing','loan','poutcome']

dsf['dummy'] = np.ones(shape = dsf.shape[0])
for col in cat_cols:
    print(col)
    counts = dsf[['dummy', 'subbed', col]].groupby(['subbed', col], as_index = False).count()
    temp = counts[counts['subbed'] == 'yes'][[col, 'dummy']]
    _ = plt.figure(figsize = (10, 4))
    plt.subplot(1,2,1)
    temp = counts[counts['subbed'] == 'yes'][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts of ' + col + '\n Subscived to term deposit')
    plt.ylabel('counts')
    plt.subplot(1,2,2)
    temp = counts[counts['subbed'] == 'no'][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('counts of' + col + '\n Has not subbed')
    plt.ylabel('counts')
    plt.show()


# # Data Prep

# In[11]:


dsf.subbed.replace(('yes', 'no'), (1, 0), inplace=True)
print(dsf.subbed)


# In[12]:


labels = np.array(dsf['subbed'])


# In[13]:


def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']

Features = encode_string(dsf['job'])
for col in categorical_columns:
    temp = encode_string(dsf[col])
    Features = np.concatenate([Features, temp], axis = 1)

print(Features.shape)
print(Features[:2, :])     


# In[14]:


Features = np.concatenate([Features, np.array(dsf[['age', 'balance', 
                            'duration', 'campaign', 'previous']])], axis = 1)
print(Features.shape)
print(Features[:2, :])   


# In[15]:


nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])


# In[16]:


scaler = preprocessing.StandardScaler().fit(X_train[:,34:])
X_train[:,34:] = scaler.transform(X_train[:,34:])
X_test[:,34:] = scaler.transform(X_test[:,34:])
X_train[:2,]


# # Regression

# In[17]:


logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(X_train, y_train)


# In[18]:


print(logistic_mod.intercept_)
print(logistic_mod.coef_)


# In[19]:


probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])


# In[20]:


def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
plot_auc(y_test, probabilities)    


# Prepared data to better scale data for clean modeling for further analysis. While utilizing features that were well defined in the data set. Data was shifted from catagorical to numerical to help better define those variables.

# In[ ]:





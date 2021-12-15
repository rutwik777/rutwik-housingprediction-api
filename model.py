#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# In[3]:


df=pd.read_csv('C:/Users/rutwi/Desktop/ML Deployment/Housing Linear Reg/housing.csv')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


sns.pairplot(df)


# In[8]:


sns.heatmap(df.corr(), annot=True)


# In[9]:


sns.jointplot(x='RM',y='MEDV',data=df)


# In[10]:


sns.lmplot(x='RM', y='MEDV',data=df)


# In[11]:


sns.lmplot(x='LSTAT', y='MEDV',data=df)


# In[18]:


sns.lmplot(x='LSTAT', y='RM',data=df)


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X=df[['RM']]
y=df[['MEDV']]


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lm=LinearRegression()


# In[17]:


lm.fit(X_train, y_train)


# In[18]:


#Save the trained model to disk
pickle.dump(lm, open('model.pkl', 'wb'))


# In[25]:


predictions=lm.predict(X_test)


# In[26]:


lm.coef_


# In[27]:


predictions


# In[28]:


plt.scatter(y_test, predictions)
plt.xlabel('Y_test')
plt.ylabel('Predictions')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv("Ecommerce Customers")


# In[3]:


customers.head()


# In[4]:


customers.describe()


# In[5]:


customers.info()


# In[6]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[7]:


# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# In[8]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# In[9]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# In[10]:


sns.pairplot(customers)


# In[11]:


# Length of Membership 


# In[12]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# In[13]:


y = customers['Yearly Amount Spent']


# In[14]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lm = LinearRegression()


# In[19]:


lm.fit(X_train,y_train)


# In[20]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# In[21]:


predictions = lm.predict( X_test)


# In[22]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[23]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[24]:


sns.distplot((y_test-predictions),bins=50);


# In[25]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:





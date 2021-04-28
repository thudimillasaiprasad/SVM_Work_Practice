#!/usr/bin/env python
# coding: utf-8

# ##SVM_Practise

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


## Lets create as small sampkle data set  frame

x = np.linspace(-5.0, 5.0, 100)
y = np.sqrt(10**2 - x**2)
y=np.hstack([y,-y])
x=np.hstack([x,-x])


# In[4]:


x1 = np.linspace(-5.0, 5.0, 100)
y1 = np.sqrt(5**2 - x1**2)
y1=np.hstack([y1,-y1])
x1=np.hstack([x1,-x1])


# In[5]:


## Lets plot the features for the graph

plt.scatter(y,x)
plt.scatter(y1,x1)


# In[ ]:


## here we see that its complex data we acheives and its 2d nature and we will convert it to 3d so we can draw a line or moduel to seperate the data.


# In[6]:


# let create a data set for the process 


df1 =pd.DataFrame(np.vstack([y,x]).T,columns=['X1','X2'])
df1['Y']=0
df2 =pd.DataFrame(np.vstack([y1,x1]).T,columns=['X1','X2'])
df2['Y']=1
df = df1.append(df2)
df.head(5)


# In[7]:


# here we have created data set.

### Independent and Dependent features
X = df.iloc[:, :2]  
y = df.Y


# In[8]:


y


# In[10]:


##  lets Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[11]:


# lets import slearn packages to classife
from sklearn.svm import SVC
classifier=SVC(kernel="rbf")
classifier.fit(X_train,y_train)


# In[12]:


## lets go to check the metrics as the data acuray is good or not

from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)


# In[13]:


df.head()


# In[15]:


## lets the see ploy nominal Equations

#k(x,y) = (x^Ty+c)^d


# In[16]:


## let go for the svc kernels
# We need to find components for the Polynomical Kernel
#X1,X2,X1_square,X2_square,X1*X2
df['X1_Square']= df['X1']**2
df['X2_Square']= df['X2']**2
df['X1*X2'] = (df['X1'] *df['X2'])
df.head()


# In[17]:


### Independent and Dependent features
X = df[['X1','X2','X1_Square','X2_Square','X1*X2']]
y = df['Y']
y


# In[18]:


## lets for train slipt

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 0)


# In[19]:


X_train


# In[20]:


## lets plot the data for the vizualsation

import plotly.express as px

fig = px.scatter_3d(df, x='X1', y='X2', z='X1*X2',
              color='Y')
fig.show()


# In[21]:


# lets plot the x and y squares and complair it how the graph vizualised

fig = px.scatter_3d(df, x='X1_Square', y='X1_Square', z='X1*X2',
              color='Y')
fig.show()


# In[22]:


## lets check for the svm classifiers for accuarcy


classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


## we draw line between as it is 


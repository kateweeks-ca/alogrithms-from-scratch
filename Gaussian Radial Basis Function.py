
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


df = pd.read_csv("knndata.csv")


# In[4]:


train = df.loc[:,['trainPoints_x1','trainPoints_x2']]
test  =df.loc[:,['testPoints_x1','testPoints_x2']]


# In[6]:


r = 4


# In[7]:


bottom = []

for i in range(0,40):
    b = (np.exp(-(((train.values - test.iloc[i,:].values)**2).sum(axis=1))/r)).sum()
    bottom.append(b)
    
bottom = np.array(bottom)


# In[9]:


top = []

for i in range(0,40):
    t = df["trainLabel"] @ (np.exp(-(((train.values - test.iloc[i,:].values)**2).sum(axis=1))/r))
    top.append(t)
    
top = np.array(top)


# In[10]:


predict = np.sign(top/bottom)


# In[12]:


print("The correct classification rate is",sum(predict == df["testLabel"])/len(predict))


# In[13]:


TP = sum((predict == 1) & (df["testLabel"] == 1))
FP = sum((predict == 1) & (df["testLabel"] == -1))
FN = sum((predict == -1) & (df["testLabel"] == 1))
TN = sum((predict == -1) & (df["testLabel"] == -1))


# In[14]:


val = ['Postive(Actual)', 'Negative(Actual)']
val2 = ['Positive(Predicted)', 'Negative(Predicted)']
data = np.array([[TP, FP],
                 [FN, TN],
                 ])

matrix = pd.DataFrame(data, val2, val)

matrix


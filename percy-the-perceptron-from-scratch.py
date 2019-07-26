
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ipywidgets import interact


# In[4]:


df = pd.read_csv("MovieData.csv")
dataset = df.values
X = df[['Genre', 'CriticsRating']].values
y = df.Watched.values
weights = np.random.normal(0,3,size=(3))


# In[5]:


#Perform EDA to make assumptions about the weights
from ipywidgets import interactive
x = np.linspace(0, 5)

def f(m=1, b=0):
    plt.plot(x, m*x + b)
    plt.ylim(0, 5)
    plt.grid(True)
    plt.axhline(0, color='k')
    plt.axvline(0, color="k")
    plt.title("$y = mx + b$:    m={} b={}".format(m,b))
    plt.scatter(df.Genre, df.CriticsRating, c=df.Watched, cmap = cm.PiYG);
    

interactive(f, m=(0, 1, .05), b=(1,4, 0.01))


# In[6]:


def predict(xi, w):
    return(np.sign(w[0] + xi @ w[1:]))


def perceptive(data, eta, epochs):
    
    weights = np.zeros(3)
    for epoch in range(epochs):
        errors = 0.0
        for inst in data:
            x = inst[:2]
            prediction = predict(x, weights)
            err = inst[2] - prediction
            errors += np.square(err)
           
            weights[0] = weights[0] + eta * err
            for i in range(len(inst)-1):
                weights[i + 1] = weights[i + 1] + eta * err * inst[i]
                    

                
        
        if(errors == 0.0000):
            break
    return weights,epoch


eta = 0.2
epochs = 1000
weights,epoch = perceptive(dataset, eta, epochs)
print("weights = ",weights)
print("epoch=", epoch)


# In[7]:


y_hat = predict(X,weights)
print(y_hat ==y)


# In[8]:


#divide by w2 to get y=1
weights=weights/weights[2]
print(weights)


# In[9]:


plt.figure(figsize=(10,5))
Xplot = np.array([1, 2, 3, 4, 5])
x = np.linspace(0, 10)
Yplot = np.array([1.1,1.9,3.0,4.1,5.2,])
plt.rcParams['axes.facecolor'] = '#c7c9cc'
plt.title("Decision Boundary")
plt.xlabel("Genre")
plt.ylabel("Critics Rating")
plt.plot(Xplot, Xplot*-weights[1] + -weights[0], 'b')
plt.legend()
plt.scatter(df.Genre, df.CriticsRating, c=df.Watched, cmap = cm.PiYG);


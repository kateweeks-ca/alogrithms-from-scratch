
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm

import warnings
warnings.filterwarnings(action="ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.metrics import pairwise_distances_argmin


def find_clusters(X, n_clusters, rseed=666):
    count = 1
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]

    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array(
            [X[labels == i].mean(0) for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all((abs(centers - new_centers) < 0.001) & (count < 1000)):
            break
        centers = new_centers
        count = count + 1
    return centers, labels, count


# In[4]:


#load in data set
dfx = pd.read_csv("rbfClassification.csv")

# convert X values to np array
X = dfx.iloc[:,0:2]
X = X.values

# convert Y values to np array
Y = dfx.iloc[:,2]
Y = Y.values


# In[5]:


centers, labels, count = find_clusters(X, 2)


# In[6]:


print(centers)


# In[7]:


fig = plt.figure(figsize=(10, 10))

colormap = np.array(['orange', 'black'])

plt.scatter(X[:, 0], X[:, 1], c=colormap[Y], s=80)
plt.title('k means for rbf')
ax = fig.add_subplot(111)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
circle1 = plt.Circle(centers[0], radius=0.2, fc='orange', alpha=0.6)
circle2 = plt.Circle(centers[1], radius=0.2, fc='black', alpha=0.6)

plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2);


# In[8]:


centers, labels, count = find_clusters(X, 2)


# In[9]:


#train an rbf model using y = .5
def k_rbf(X, Y, centers, gamma):
    phi = np.zeros(shape=(len(X), 3))
    phi[:, 0] = np.ones((len(X), ), dtype=int)
    K = 2
    for k in range(K):
        phi[:, k + 1] = np.exp(-gamma * ((X - centers[k])**2)).sum(1)
    W = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    return (W, phi)


# In[10]:


W, phi = k_rbf(X, Y, centers, .5)


# In[11]:


#Report Classification Rate


# In[12]:


y_predict = phi @ W
y_predict = np.round(y_predict)
accuracy = (y_predict==Y).sum()/len(y_predict)
print("The accuracy rate is: ", accuracy * 100)


# In[13]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, random_state=13)
km.fit(X)
y_km = km.predict(X)
mu_km = km.cluster_centers_


# In[15]:


print(mu_km)


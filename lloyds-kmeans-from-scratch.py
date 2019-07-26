
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


# load the data set
kmeans = pd.read_csv("kMeansData.csv")
kmeans.shape

kmeans.head()


# In[3]:


# perform EDA using scatter plot
fig = plt.figure(figsize=(10, 10))

plt.scatter(kmeans.iloc[:, 0], kmeans.iloc[:, 1], c="Orange", s=80)
plt.title('k-means data set')
ax = fig.add_subplot(111)
ax.set_xlabel('x1')
ax.set_ylabel('x2');


# In[4]:


def kmu(X, n):

    # create empty arrays for the new centroid values and the difference
    new_centroids = np.zeros(shape=(3, 2))
    diff = np.zeros(shape=(3, 2))

    # initalize our centers using random data points from our data
    rseed = 415
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:3]
    centroids = X.iloc[i, 0:2].values

    # initalize empty columns for distances
    X['Association'] = pd.Series(np.zeros(X.shape[0]), index=X.index)
    X['Dist_C1'] = pd.Series(np.zeros(X.shape[0]), index=X.index)
    X['Dist_C2'] = pd.Series(np.zeros(X.shape[0]), index=X.index)
    X['Dist_C3'] = pd.Series(np.zeros(X.shape[0]), index=X.index)

    # loop over the lloyd's alog. 1000 times
    for count in range(1, n):
        X["Dist_C1"] = np.sqrt((X.iloc[:, 0] - centroids[0, 0])**2 +
                               (X.iloc[:, 1] - centroids[0, 1])**2)
        X["Dist_C2"] = np.sqrt((X.iloc[:, 0] - centroids[1, 0])**2 +
                               (X.iloc[:, 1] - centroids[1, 1])**2)
        X["Dist_C3"] = np.sqrt((X.iloc[:, 0] - centroids[2, 0])**2 +
                               (X.iloc[:, 1] - centroids[2, 1])**2)

        X["Association"] = np.where(
            (X.Dist_C1 < X.Dist_C2) & (X.Dist_C1 < X.Dist_C3), 1,
            np.where((X.Dist_C2 < X.Dist_C1) & (X.Dist_C2 < X.Dist_C3), 2, 3))

        new_centroids[0, :] = X[X.Association == 1][["x1", "x2"]].mean()
        new_centroids[1, :] = X[X.Association == 2][["x1", "x2"]].mean()
        new_centroids[2, :] = X[X.Association == 3][["x1", "x2"]].mean()

        diff = abs(centroids - new_centroids)

        #break if difference between new & old centroids is > .001
        if diff.all() < .001:
            break

        else:
            new_centroids = centroids

        count = count + 1
    return (new_centroids, count, diff)


# In[5]:


#run the function
centroids, count, diff = kmu(kmeans, 1000)


# In[6]:


#report the clusters
print(centroids)


# In[7]:


#check the count
print(count)


# In[8]:


fig = plt.figure(figsize=(10, 10))

colormap = np.array(['black', 'purple', 'green', 'blue'])

plt.scatter(kmeans.x1, kmeans.x2, c=colormap[kmeans.Association], s=80)
plt.title('k means')
ax = fig.add_subplot(111)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
circle1 = plt.Circle(centroids[0, :], radius=0.2, fc='purple', alpha=0.4)
circle2 = plt.Circle(centroids[1, :], radius=0.2, fc='green', alpha=0.4)
circle3 = plt.Circle(centroids[2, :], radius=0.2, fc='blue', alpha=0.4)
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3);


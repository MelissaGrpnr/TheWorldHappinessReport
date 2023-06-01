#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# In[31]:


data= pd.read_csv('world-happiness-report-2021.csv')
data.head()


# In[32]:


# Select features for clustering
features = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices',
            'Generosity', 'Perceptions of corruption']

X = data[features]


# In[33]:


# Preprocess the data by scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters
n_clusters = 3

# Fit the K-means model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Evaluate the clustering results
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Average silhouette score: {silhouette_avg}")

# Analyze and interpret the clusters
data['Cluster'] = cluster_labels
cluster_means = data.groupby('Cluster')[features].mean()
print(cluster_means)


# In[34]:


# Visualize the clusters using scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Logged GDP per capita', y='Healthy life expectancy', hue='Cluster', palette='Set1')
plt.title('Clustering Results')
plt.xlabel('Logged GDP per capita')
plt.ylabel('Healthy life expectancy')
plt.show()


# In[35]:


# Visualize the clusters using pairplot
sns.pairplot(data=data, vars=features, hue='Cluster', palette='Set1')
plt.suptitle('Clustering Results - Pairplot')
plt.show()


# In[36]:


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Logged GDP per capita'], data['Social support'], data['Healthy life expectancy'],
           c=cluster_labels, cmap='Set1')
ax.set_xlabel('Logged GDP per capita')
ax.set_ylabel('Social support')
ax.set_zlabel('Healthy life expectancy')
ax.set_title('Clustering Results - 3D Scatter Plot')
plt.show()


# In[ ]:





# In[ ]:





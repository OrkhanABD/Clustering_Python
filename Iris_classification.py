#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target


# In[5]:


# Applying K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data.iloc[:, :-1])


# In[6]:


# Scatter plot for K-Means Clustering
plt.figure(figsize=(10, 6))
plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=data['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering on Iris Dataset (Sepal Length vs Sepal Width)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(label='Cluster')
plt.show()


# In[7]:


# Applying Hierarchical Clustering
hier_clustering = AgglomerativeClustering(n_clusters=3)
data['Hierarchical_Cluster'] = hier_clustering.fit_predict(data.iloc[:, :-2])


# In[8]:


# Scatter plot for Hierarchical Clustering
plt.figure(figsize=(10, 6))
plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=data['Hierarchical_Cluster'], cmap='plasma')
plt.title('Hierarchical Clustering on Iris Dataset (Sepal Length vs Sepal Width)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(label='Cluster')
plt.show()


# In[9]:


# Plotting Dendrogram for Hierarchical Clustering
plt.figure(figsize=(10, 7))
Z = linkage(data.iloc[:, :-2], method='ward')
dendrogram(Z, truncate_mode='level', p=3, show_leaf_counts=True)
plt.title('Dendrogram for Iris Dataset (Hierarchical Clustering)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()


# In[ ]:





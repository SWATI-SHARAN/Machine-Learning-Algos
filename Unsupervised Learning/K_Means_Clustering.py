import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs #Generates synthetic cluster data.
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

kmeans_model = KMeans(n_clusters=4, random_state=42) #Tells KMeans to find 4 clusters

kmeans_model.fit(X)
cluster_centers = kmeans_model.cluster_centers_
labels = kmeans_model.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k',s=30) #The first scatter plots data points, colored by their predicted cluster (c=labels).


plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')# The second scatter plots the centroids as big red "X" marks.
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend() #Activates the legend and displays the label 'Cluster Centers' in the plot area.
plt.show()
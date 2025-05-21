from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs 

X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

linkage_matrix = linkage(X, method='ward') #method='ward': Minimizes the variance within each cluster 

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, 
           orientation='top', 
           labels=y, 
           distance_sort='descending', 
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

clusters = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_pred = clusters.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis') #plt.scatter() --> Visualize resulting clusters in 2D
plt.title('Agglomerative Hierarchical Clustering')
plt.show()
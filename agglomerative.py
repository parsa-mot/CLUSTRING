import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# 1) Create a Dendrogram
linked = linkage(X_scaled, method='ward')  # 'ward', 'complete', 'average', 'single'
plt.figure(figsize=(12, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()

# 2) Fit Agglomerative Clustering
n_clusters = 3  # choose based on dendrogram
agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
df['cluster'] = agg_cluster.fit_predict(X_scaled)


# 3) calculate Silhouette Score for Agglomerative Clustering
score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)

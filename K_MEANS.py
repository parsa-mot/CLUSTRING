# 1) Basic K-means from sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Train K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(df)
df["cluster"] = kmeans.predict(df)

# Cluster centers and labels
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)

# Plot
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            c="red", marker="x", s=200, label="Centroids")
plt.legend()
plt.show()

# 1) Basic K-means from sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Train K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(df)
df["cluster"] = kmeans.predict(df) #creating a separate col for clusters

1-1) plotting the clusters
from sklearn.decomposition import PCA
# Reduce to 2D for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(df[float_cols])

plt.scatter(
    reduced[:, 0], reduced[:, 1],
    c=df["cluster"], cmap="viridis", s=50
)

# Plot PCA-reduced cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centers_pca[:, 0], centers_pca[:, 1],
    c="red", s=200, alpha=0.75, marker="X"
)

plt.title("K-means Clustering (PCA-reduced)")
plt.show()



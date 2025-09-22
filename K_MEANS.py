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




# 2) elbow method
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

inertia = []
K = range(1, 20)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[float_cols])
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, "bo-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")

# Force all k values on x-axis
plt.xticks(np.arange(1, 20, 1))

# Add vertical dotted lines
for k, val in zip(K, inertia):
    plt.vlines(x=k, ymin=0, ymax=val, colors="gray", linestyles="dotted", alpha=0.6)

plt.show()

plt.scatter(
    centers_pca[:, 0], centers_pca[:, 1],
    c="red", s=200, alpha=0.75, marker="X"
)

plt.title("K-means Clustering (PCA-reduced)")
plt.show()



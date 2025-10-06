from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X, labels)

from sklearn.metrics import davies_bouldin_score
DBI_Score = davies_bouldin_score(X, labels)

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(X, labels)



print(f"Silhouette Score: {sil_score:.3f}")
print(f"DBI Score: {DBI_Score:.3f}")
print(f"CH Score: {ch:.3f}")

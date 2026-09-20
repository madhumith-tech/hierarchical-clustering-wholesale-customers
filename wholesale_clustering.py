import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

df = pd.read_csv("Wholesale customers data.csv")

print("Dataset shape:", df.shape)
print(df.head())


# ---------------------------------------------------------
# 2. Select product category columns
# ---------------------------------------------------------

features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[features]


# ---------------------------------------------------------
# 3. Standardize the data
# ---------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# 4. Calculate original pairwise distances
#    Used for Cophenetic Correlation
# ---------------------------------------------------------

original_distance = pdist(X_scaled, metric="euclidean")


# ---------------------------------------------------------
# 5. Create linkage matrices
# ---------------------------------------------------------

ward_linkage = linkage(X_scaled, method="ward")
complete_linkage = linkage(X_scaled, method="complete")
average_linkage = linkage(X_scaled, method="average")


# ---------------------------------------------------------
# 6. Plot dendrograms
# ---------------------------------------------------------

plt.figure(figsize=(15, 6))

dendrogram(
    ward_linkage,
    truncate_mode="lastp",
    p=30
)

plt.title("Ward Linkage Dendrogram")
plt.xlabel("Customers / Cluster")
plt.ylabel("Distance")
plt.show()


plt.figure(figsize=(15, 6))

dendrogram(
    complete_linkage,
    truncate_mode="lastp",
    p=30
)

plt.title("Complete Linkage Dendrogram")
plt.xlabel("Customers / Cluster")
plt.ylabel("Distance")
plt.show()


plt.figure(figsize=(15, 6))

dendrogram(
    average_linkage,
    truncate_mode="lastp",
    p=30
)

plt.title("Average Linkage Dendrogram")
plt.xlabel("Customers / Cluster")
plt.ylabel("Distance")
plt.show()


# ---------------------------------------------------------
# 7. Function to calculate Cophenetic Correlation
# ---------------------------------------------------------

def calculate_cophenetic(Z):
    coefficient, _ = cophenet(
        Z,
        original_distance
    )

    return coefficient


ward_coph = calculate_cophenetic(ward_linkage)
complete_coph = calculate_cophenetic(complete_linkage)
average_coph = calculate_cophenetic(average_linkage)


# ---------------------------------------------------------
# 8. Function for clustering and Silhouette Score
# ---------------------------------------------------------

def evaluate_clustering(Z, name):

    print("\n", name)
    print("-" * 40)

    for k in range(2, 7):

        labels = fcluster(
            Z,
            k,
            criterion="maxclust"
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        print(
            "Clusters:",
            k,
            "Silhouette Score:",
            round(score, 4)
        )


evaluate_clustering(ward_linkage, "WARD")
evaluate_clustering(complete_linkage, "COMPLETE")
evaluate_clustering(average_linkage, "AVERAGE")


# ---------------------------------------------------------
# 9. Final clustering using 2 clusters
# ---------------------------------------------------------

ward_labels = fcluster(
    ward_linkage,
    2,
    criterion="maxclust"
)

complete_labels = fcluster(
    complete_linkage,
    2,
    criterion="maxclust"
)

average_labels = fcluster(
    average_linkage,
    2,
    criterion="maxclust"
)


# ---------------------------------------------------------
# 10. Compare cluster assignments
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Customer": range(1, len(df) + 1),
    "Ward": ward_labels,
    "Complete": complete_labels,
    "Average": average_labels
})

print("\nCluster Assignment Comparison")
print(comparison.head(20))


# ---------------------------------------------------------
# 11. Cluster sizes
# ---------------------------------------------------------

print("\nWard Cluster Sizes:")
print(pd.Series(ward_labels).value_counts().sort_index())

print("\nComplete Cluster Sizes:")
print(pd.Series(complete_labels).value_counts().sort_index())

print("\nAverage Cluster Sizes:")
print(pd.Series(average_labels).value_counts().sort_index())


# ---------------------------------------------------------
# 12. Evaluation summary
# ---------------------------------------------------------

print("\n==============================")
print("FINAL EVALUATION")
print("==============================")

print(
    "Ward Cophenetic Correlation:",
    round(ward_coph, 4)
)

print(
    "Complete Cophenetic Correlation:",
    round(complete_coph, 4)
)

print(
    "Average Cophenetic Correlation:",
    round(average_coph, 4)
)

print(
    "Ward Silhouette Score:",
    round(
        silhouette_score(X_scaled, ward_labels),
        4
    )
)

print(
    "Complete Silhouette Score:",
    round(
        silhouette_score(X_scaled, complete_labels),
        4
    )
)

print(
    "Average Silhouette Score:",
    round(
        silhouette_score(X_scaled, average_labels),
        4
    )
)
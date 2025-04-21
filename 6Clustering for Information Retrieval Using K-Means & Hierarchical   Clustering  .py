# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups  # Load 20 Newsgroups dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to TF-IDF features
from sklearn.cluster import KMeans, AgglomerativeClustering  # K-Means & Hierarchical clustering
from sklearn.metrics import silhouette_score  # Evaluate clustering performance
from scipy.cluster.hierarchy import dendrogram, linkage  # Hierarchical clustering visualization

# Load a subset of 20 Newsgroups dataset (4 categories for simplicity)
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Convert text data into numerical format using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # Limit features for efficiency
X = vectorizer.fit_transform(newsgroups.data)  # Transform text into TF-IDF vectors

# ---------------- K-MEANS CLUSTERING ----------------
# Define number of clusters (same as number of categories)
num_clusters = len(categories)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)  # Assign clusters

# Compute silhouette score to evaluate clustering performance
silhouette_avg = silhouette_score(X, kmeans_labels)

# Display K-Means results
print("\n=== K-Means Clustering Results ===")
print(f"Number of Clusters: {num_clusters}")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# ---------------- HIERARCHICAL CLUSTERING ----------------
# Perform Agglomerative Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward') # Remove affinity parameter
hierarchical_labels = hierarchical.fit_predict(X.toarray())  # Convert sparse matrix to array

# Plot dendrogram for hierarchical clustering
plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendrogram")
Z = linkage(X.toarray(), method='ward')  # Compute hierarchical clustering linkage matrix
dendrogram(Z, truncate_mode="level", p=10)  # Visualize only first few levels
plt.show()

# Display Hierarchical clustering results
print("\n=== Hierarchical Clustering Results ===")
print(f"Number of Clusters: {num_clusters}")
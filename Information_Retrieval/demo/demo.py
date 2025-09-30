# demo.py
# Small hands-on project: User embeddings + clustering + visualization + temporal weighting
# Dataset: Online Retail (https://archive.ics.uci.edu/ml/datasets/online+retail)

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")

# Basic cleaning
df = df.dropna(subset=["CustomerID", "Description", "InvoiceDate"])
df = df[df["Quantity"] > 0]

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])  # 去掉无法转换的

# Step 2. Add recency (time decay factor)
print("Adding temporal features...")
latest_date = df["InvoiceDate"].max()

# 计算 recency = 距离最后交易的天数
df["RecencyDays"] = (latest_date - df["InvoiceDate"]).dt.days
# 定义一个衰减权重（越近的交易权重越大）
df["RecencyWeight"] = np.exp(-df["RecencyDays"] / 180)  # 半年为衰减尺度

# Step 3. Build user sequences (list of product descriptions per user)
print("Building user sequences...")
user_sequences = df.groupby("CustomerID")["Description"].apply(list)

# Step 4. Train Word2Vec model on product sequences
print("Training Word2Vec embeddings...")
sentences = user_sequences.tolist()
w2v = Word2Vec(sentences, vector_size=50, window=5, min_count=5, workers=4, seed=42)

# Step 5. Compute user embeddings with temporal weighting
print("Computing user embeddings with recency weights...")

def get_user_vector(user_id, products):
    subset = df[df["CustomerID"] == user_id]
    vectors, weights = [], []
    for _, row in subset.iterrows():
        p = row["Description"]
        if p in w2v.wv:
            vectors.append(w2v.wv[p])
            weights.append(row["RecencyWeight"])
    if vectors:
        return np.average(vectors, axis=0, weights=weights)
    else:
        return np.zeros(w2v.vector_size)

user_embeddings = user_sequences.index.to_series().apply(
    lambda uid: get_user_vector(uid, user_sequences[uid])
)
X = np.vstack(user_embeddings.values)

# Step 6. Clustering (K-means)
print("Clustering users...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Step 7. Visualization (t-SNE)
print("Running t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
plt.title("User Clusters (with Temporal Weighting)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar()
plt.show()

# Step 8. Print top products per cluster
print("\nTop products in each cluster:")
df["cluster"] = df["CustomerID"].map(dict(zip(user_sequences.index, labels)))
for c in sorted(df["cluster"].unique()):
    top_products = (
        df[df["cluster"] == c]["Description"]
        .value_counts()
        .head(3)
        .index.tolist()
    )
    print(f"Cluster {c}: {top_products}")

# Step 9. Save results to CSV
print("\nSaving results to 'user_clusters.csv' ...")
user_cluster_df = pd.DataFrame({
    "CustomerID": user_sequences.index,
    "Cluster": labels
})

# 选取每个用户的平均 Recency 权重，方便分析
recency_mean = df.groupby("CustomerID")["RecencyWeight"].mean()
user_cluster_df["AvgRecencyWeight"] = user_cluster_df["CustomerID"].map(recency_mean)

# 保存 CSV 文件
user_cluster_df.to_csv("user_clusters.csv", index=False)
print("Saved user_clusters.csv with columns: CustomerID, Cluster, AvgRecencyWeight")



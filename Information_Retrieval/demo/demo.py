# demo_multimodel.py
# Enhanced version: Compare multiple user embedding and clustering approaches
# Author: Tobias Liu (for JET User Behaviour Analytics Intern demo)

import pandas as pd
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. Data Loading & Cleaning
# ------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")
df = df.dropna(subset=["CustomerID", "Description", "InvoiceDate"])
df = df[df["Quantity"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])

latest_date = df["InvoiceDate"].max()
df["RecencyDays"] = (latest_date - df["InvoiceDate"]).dt.days
df["RecencyWeight"] = np.exp(-df["RecencyDays"] / 180)

# 用户购买序列
user_sequences = df.groupby("CustomerID")["Description"].apply(list)

# ------------------------------------------------------------
# 2. Embedding Functions
# ------------------------------------------------------------
def get_word2vec_embeddings(sentences, user_sequences, df):
    print("Training Word2Vec...")
    w2v = Word2Vec(sentences, vector_size=50, window=5, min_count=5, workers=4, seed=42)

    def get_user_vector(user_id):
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

    user_embeddings = user_sequences.index.to_series().apply(get_user_vector)
    return np.vstack(user_embeddings.values)

def get_doc2vec_embeddings(user_sequences):
    print("Training Doc2Vec...")
    documents = [TaggedDocument(words=seq, tags=[str(uid)]) for uid, seq in user_sequences.items()]
    d2v = Doc2Vec(documents, vector_size=50, window=5, min_count=5, workers=4, seed=42)
    return np.vstack([d2v.dv[str(uid)] for uid in user_sequences.index])

def get_tfidf_embeddings(user_sequences):
    print("Building TF-IDF + PCA embeddings...")
    corpus = [" ".join(seq) for seq in user_sequences]
    tfidf = TfidfVectorizer(max_features=2000)
    tfidf_matrix = tfidf.fit_transform(corpus)
    pca = PCA(n_components=50, random_state=42)
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    return reduced

# ------------------------------------------------------------
# 3. Clustering + Evaluation
# ------------------------------------------------------------
def evaluate_clustering(X, labels, model_name):
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
    ch = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else np.nan
    db = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan
    return {"Model": model_name, "Silhouette": sil, "Calinski": ch, "DaviesBouldin": db}

def clustering_models(X):
    results = []
    models = {
        "KMeans": KMeans(n_clusters=5, random_state=42, n_init=10),
        "GMM": GaussianMixture(n_components=5, random_state=42),
        "HDBSCAN": HDBSCAN(min_cluster_size=15)
    }
    for name, model in models.items():
        print(f"Clustering with {name}...")
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels, name)
        results.append(metrics)
        visualize_tsne(X, labels, title=f"{name} Clusters (2D projection)")
    return pd.DataFrame(results)

# ------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------
def visualize_tsne(X, labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=800)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()

# ------------------------------------------------------------
# 5. Run Embedding Comparisons
# ------------------------------------------------------------
sentences = user_sequences.tolist()

embedding_methods = {
    "Word2Vec": get_word2vec_embeddings(sentences, user_sequences, df),
    "Doc2Vec": get_doc2vec_embeddings(user_sequences),
    "TFIDF_PCA": get_tfidf_embeddings(user_sequences),
}

all_results = []

for name, X in embedding_methods.items():
    print(f"\n===== Evaluating {name} embeddings =====")
    df_result = clustering_models(X)
    df_result["Embedding"] = name
    all_results.append(df_result)

final_eval = pd.concat(all_results, ignore_index=True)
print("\n=== Final Comparison Summary ===")
print(final_eval)

# ------------------------------------------------------------
# 6. Cluster Profiling (business relevance)
# ------------------------------------------------------------
# Example: using last used labels (from last model run)
labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(embedding_methods["Word2Vec"])
df["cluster"] = df["CustomerID"].map(dict(zip(user_sequences.index, labels)))

print("\nTop products per cluster:")
for c in sorted(df["cluster"].unique()):
    top_products = df[df["cluster"] == c]["Description"].value_counts().head(3).index.tolist()
    avg_recency = df[df["cluster"] == c]["RecencyDays"].mean()
    print(f"Cluster {c}: {top_products} | Avg Recency = {avg_recency:.1f} days")

# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------
final_eval.to_csv("embedding_clustering_comparison.csv", index=False)
print("\nSaved embedding_clustering_comparison.csv with metrics for all models.")

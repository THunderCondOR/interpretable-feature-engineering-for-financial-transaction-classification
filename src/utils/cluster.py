from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import hashlib

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_input_prefix(model_name: str) -> str:
    """Return the model-prescribed prefix for non-retrieval feature tasks."""
    if model_name.startswith("intfloat/multilingual-e5"):
        return "query: "
    return ""


def embed_texts(texts, model_name="tf-idf"):
    if model_name == "tf-idf":
        # Fallback: simple TF-IDF based embeddings
        print("Using fallback TF-IDF embeddings...")
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=384, stop_words=None, lowercase=True)
        embeddings = vectorizer.fit_transform(texts).toarray()

        # Normalize to unit vectors (similar to SentenceTransformer's normalize_embeddings=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms

        print(f"Generated TF-IDF embeddings with shape: {embeddings.shape}")
        return embeddings

    print("Encoding texts with SentenceTransformer...")
    model_kwargs = {
        "device": device,
        "trust_remote_code": model_name.startswith("Alibaba-NLP/"),
    }
    try:
        # Resumed runs must not require a fresh metadata request for a model
        # that is already present in the Hugging Face cache.
        model = SentenceTransformer(
            model_name,
            local_files_only=True,
            **model_kwargs,
        )
    except OSError:
        # Permit the normal download path only for a genuinely new model.
        model = SentenceTransformer(model_name, **model_kwargs)
    prefix = embedding_input_prefix(model_name)
    prepared = [f"{prefix}{text}" for text in texts]
    return np.asarray(model.encode(
        prepared, batch_size=256,
        normalize_embeddings=True, show_progress_bar=True,
    ))

def fine_cluster_dbscan(embeddings, eps=0.1, min_samples=2):
    """
    Fine-grained clustering using DBSCAN with cosine distance.
    Outliers (-1) are reassigned into unique singleton clusters.
    
    embeddings: np.array of shape (n_samples, emb_dim)
    eps: float, neighborhood distance threshold (smaller = finer clusters)
    min_samples: int, minimum samples to form a cluster

    Returns:
        cluster_ids: np.array of shape (n_samples,)
    """
    
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine"
    )
    cluster_ids = clusterer.fit_predict(embeddings)

    # Convert outliers (-1) to singleton clusters
    # max_id = cluster_ids.max()
    # for i in range(len(cluster_ids)):
    #     if cluster_ids[i] == -1:
    #         max_id += 1
    #         cluster_ids[i] = max_id

    print(f"Assigned {cluster_ids.max() + 1} clusters.")
    return cluster_ids

def fine_cluster_agglomerative(embeddings, distance_threshold=0.01):
    """
    Fine-grained clustering using AgglomerativeClustering with a distance threshold.
    All points are assigned to a cluster.
    
    embeddings: np.array of shape (n_samples, emb_dim)
    distance_threshold: float, maximum distance to merge clusters (smaller = finer clusters)

    Returns:
        cluster_ids: np.array of shape (n_samples,)
    """
    
    clusterer = AgglomerativeClustering(
        n_clusters=None,                # determined by threshold
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )
    cluster_ids = clusterer.fit_predict(embeddings)

    n_clusters = cluster_ids.max() + 1
    print(f"Assigned {n_clusters} clusters.")
    return cluster_ids

def fixed_cluster_agglomerative(embeddings, n_clusters=10):
    """
    Agglomerative clustering that creates exactly n clusters.
    All points are assigned to one of the n clusters.

    Args:
        embeddings (np.ndarray): shape (n_samples, emb_dim)
        n_clusters (int): desired number of clusters

    Returns:
        cluster_ids (np.ndarray): shape (n_samples,)
    """

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average"
    )

    cluster_ids = clusterer.fit_predict(embeddings)
    print(f"Assigned exactly {n_clusters} clusters.")
    return cluster_ids

def compute_tsne(X, PCA_n_components = 50, TSNE_n_components=2, perplexity=15, random_state=42):
    X_scaled = StandardScaler().fit_transform(X)
    X_PCA = PCA(n_components=min(PCA_n_components, X_scaled.shape[0], X_scaled.shape[1]), random_state=random_state).fit_transform(X_scaled)
    tsne = TSNE(
        n_components=TSNE_n_components, 
        perplexity=perplexity, 
        learning_rate='auto',
        init='pca',
        random_state=random_state)
    X_tsne = tsne.fit_transform(X_PCA)
    return X_tsne

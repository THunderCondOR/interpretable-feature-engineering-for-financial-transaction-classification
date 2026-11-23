"""Encode claim texts for the semantic feature space."""

import numpy as np


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

    import torch
    from sentence_transformers import SentenceTransformer

    print("Encoding texts with SentenceTransformer...")
    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
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
    return np.asarray(
        model.encode(
            prepared,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )

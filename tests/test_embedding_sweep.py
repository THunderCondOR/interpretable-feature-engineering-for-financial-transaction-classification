import numpy as np

from src.utils import cluster
from src.utils.cluster import embedding_input_prefix
from scripts.run_rosbank_embedding_sweep import EMBEDDERS, metric_means


def test_embedding_protocol_uses_e5_clustering_prefix_only():
    assert embedding_input_prefix(EMBEDDERS["multilingual_e5_large"]) == "query: "
    assert embedding_input_prefix(EMBEDDERS["gte_multilingual_base"]) == ""
    assert embedding_input_prefix(EMBEDDERS["bge_m3"]) == ""


def test_embedding_sweep_has_three_distinct_models():
    assert len(EMBEDDERS) == 3
    assert len(set(EMBEDDERS.values())) == 3


def test_sentence_transformer_prefers_local_cache(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(cluster, "SentenceTransformer", FakeModel)
    cluster.embed_texts(["claim"], "intfloat/multilingual-e5-large")
    assert calls[0][1]["local_files_only"] is True
    assert len(calls) == 1


def test_sentence_transformer_downloads_only_when_cache_missing(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))
            if kwargs.get("local_files_only"):
                raise OSError("not cached")

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(cluster, "SentenceTransformer", FakeModel)
    cluster.embed_texts(["claim"], "new/model")
    assert calls[0][1]["local_files_only"] is True
    assert "local_files_only" not in calls[1][1]

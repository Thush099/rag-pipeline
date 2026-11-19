"""Unit tests for BM25Retriever and DenseRetriever."""

import math
import numpy as np
import pytest

from src.models import BM25Retriever, DenseRetriever


def test_bm25_idf_ordering_rare_exceeds_common() -> None:
    corpus = [
        {"id": "d1", "title": "Intro", "text": "neural search uses tokens and vectors"},
        {"id": "d2", "title": "Scale", "text": "neural search scales across shards"},
        {"id": "d3", "title": "Cache", "text": "neural search caches query embeddings"},
        {"id": "d4", "title": "Rare", "text": "neural okapi bm25 uses k1=1.5"},
    ]
    retriever = BM25Retriever(corpus, k1=1.5, b=0.75)

    # "neural" appears in 4/4 docs, "search" in 3/4 docs, "okapi" in 1/4 docs
    assert retriever.idf("okapi") > retriever.idf("search") > retriever.idf("neural")
    assert retriever.idf("nonexistent") == 0.0

    # Verify exact smoothed IDF formula for "okapi" (df=1, N=4)
    expected_okapi_idf = math.log(1.0 + (4 - 1 + 0.5) / (1 + 0.5))
    assert math.isclose(retriever.idf("okapi"), expected_okapi_idf, rel_tol=1e-9)


def test_bm25_relevance_and_length_normalization() -> None:
    corpus = [
        {
            "id": "concise",
            "title": "BM25",
            "text": "Okapi BM25 scores documents with k1 and b parameters.",
        },
        {
            "id": "verbose",
            "title": "Overview",
            "text": (
                "BM25 is mentioned once here alongside many unrelated filler words "
                "about databases networks containers compilers kernels storage "
                "routers schedulers telemetry queues buffers caches monitors."
            ),
        },
        {
            "id": "irrelevant",
            "title": "Cooking",
            "text": "Bake sourdough bread at high heat with steam.",
        },
    ]
    retriever = BM25Retriever(corpus)
    results = retriever.retrieve("Okapi BM25 parameters", top_k=3)

    assert [r.doc_id for r in results] == ["concise", "verbose"]
    assert results[0].score > results[1].score > 0.0
    assert results[0].bm25_score == results[0].score
    assert results[0].rank == 1
    assert results[0].source == "bm25"


def test_dense_retriever_tfidf_cosine_and_l2_normalization() -> None:
    corpus = [
        {"id": "d1", "title": "RRF", "text": "Reciprocal Rank Fusion combines ranked retrieval lists."},
        {"id": "d2", "title": "HNSW", "text": "Hierarchical navigable small world graphs index vectors."},
        {"id": "d3", "title": "Chunking", "text": "Sliding window chunking keeps context overlap."},
    ]
    retriever = DenseRetriever(corpus)

    # All non-zero corpus embeddings must be unit L2 norm
    norms = np.linalg.norm(retriever.doc_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    results = retriever.retrieve("How does Reciprocal Rank Fusion merge lists?", top_k=2)
    assert len(results) >= 1
    assert results[0].doc_id == "d1"
    assert results[0].dense_score == pytest.approx(results[0].score)
    assert 0.0 < results[0].score <= 1.0 + 1e-6


def test_dense_retriever_rejects_document_embeddings_with_wrong_row_count() -> None:
    corpus = [
        {"id": "d1", "title": "Alpha", "text": "first doc"},
        {"id": "d2", "title": "Beta", "text": "second doc"},
    ]

    class InvalidBackend:
        def encode_documents(self, texts):
            return np.ones((1, 2), dtype=np.float32)

        def encode_query(self, query: str):
            return np.ones(2, dtype=np.float32)

    with pytest.raises(ValueError, match="one embedding per document"):
        DenseRetriever(corpus, embedding_backend=InvalidBackend())


def test_dense_retriever_rejects_non_finite_document_embeddings() -> None:
    corpus = [{"id": "d1", "title": "Alpha", "text": "first doc"}]

    class InvalidBackend:
        def encode_documents(self, texts):
            return np.array([[np.nan, 1.0]], dtype=np.float32)

        def encode_query(self, query: str):
            return np.ones(2, dtype=np.float32)

    with pytest.raises(ValueError, match="finite values"):
        DenseRetriever(corpus, embedding_backend=InvalidBackend())


def test_dense_retriever_rejects_duplicate_document_ids() -> None:
    corpus = [
        {"id": "duplicate", "title": "Alpha", "text": "first doc"},
        {"id": "duplicate", "title": "Beta", "text": "second doc"},
    ]

    with pytest.raises(ValueError, match="unique document IDs"):
        DenseRetriever(corpus)


def test_dense_retriever_rejects_query_embedding_with_wrong_dimension() -> None:
    corpus = [{"id": "d1", "title": "Alpha", "text": "first doc"}]

    class InvalidBackend:
        def encode_documents(self, texts):
            return np.ones((1, 2), dtype=np.float32)

        def encode_query(self, query: str):
            return np.ones(3, dtype=np.float32)

    retriever = DenseRetriever(corpus, embedding_backend=InvalidBackend())
    with pytest.raises(ValueError, match="same dimension"):
        retriever.get_scores("query")


def test_callable_embedding_backend_rejects_multiple_query_embeddings() -> None:
    corpus = [{"id": "d1", "title": "Alpha", "text": "first doc"}]

    def encoder(texts):
        if len(texts) == 1 and texts[0] == "query":
            return np.ones((2, 2), dtype=np.float32)
        return np.ones((len(texts), 2), dtype=np.float32)

    retriever = DenseRetriever(corpus, embedding_backend=encoder)
    with pytest.raises(ValueError, match="one query embedding"):
        retriever.get_scores("query")


def test_dense_retriever_rejects_multiple_query_embeddings() -> None:
    corpus = [{"id": "d1", "title": "Alpha", "text": "first doc"}]

    class InvalidBackend:
        def encode_documents(self, texts):
            return np.ones((1, 2), dtype=np.float32)

        def encode_query(self, query: str):
            return np.ones((2, 2), dtype=np.float32)

    retriever = DenseRetriever(corpus, embedding_backend=InvalidBackend())
    with pytest.raises(ValueError, match="one query embedding"):
        retriever.get_scores("query")


def test_dense_retriever_rejects_non_finite_query_embeddings() -> None:
    corpus = [{"id": "d1", "title": "Alpha", "text": "first doc"}]

    class InvalidBackend:
        def encode_documents(self, texts):
            return np.ones((1, 2), dtype=np.float32)

        def encode_query(self, query: str):
            return np.array([np.inf, 1.0], dtype=np.float32)

    retriever = DenseRetriever(corpus, embedding_backend=InvalidBackend())
    with pytest.raises(ValueError, match="finite values"):
        retriever.get_scores("query")


def test_dense_retriever_accepts_protocol_and_callable_backends() -> None:
    corpus = [
        {"id": "d1", "title": "Alpha", "text": "first doc"},
        {"id": "d2", "title": "Beta", "text": "second doc"},
    ]

    class MockProtocolBackend:
        def encode_documents(self, texts):
            return np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

        def encode_query(self, query: str):
            return np.array([0.0, 5.0], dtype=np.float32)

    proto_retriever = DenseRetriever(corpus, embedding_backend=MockProtocolBackend())
    proto_results = proto_retriever.retrieve("anything", top_k=2)
    assert proto_results[0].doc_id == "d2"
    assert proto_results[0].score == pytest.approx(1.0, abs=1e-5)
    assert proto_results[1].score == pytest.approx(0.0, abs=1e-5)

    def callable_encoder(texts):
        vectors = []
        for text in texts:
            if "semantic" in text or "first" in text:
                vectors.append([3.0, 0.0])
            else:
                vectors.append([0.0, 3.0])
        return np.array(vectors, dtype=np.float32)

    callable_retriever = DenseRetriever(corpus, embedding_backend=callable_encoder)
    callable_results = callable_retriever.retrieve("semantic query", top_k=1)
    assert callable_results[0].doc_id == "d1"
    assert callable_results[0].score == pytest.approx(1.0, abs=1e-5)

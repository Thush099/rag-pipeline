"""Unit tests for BM25Retriever and DenseRetriever."""

import math
from pathlib import Path

import numpy as np
import pytest

from src.models import (
    BM25Retriever,
    CrossEncoderReRanker,
    DenseRetriever,
    ReciprocalRankFusion,
    TfidfEmbeddingBackend,
)
from src.pipeline import RAGPipeline
from src.utils import load_corpus

SAMPLE_CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.jsonl"


@pytest.mark.parametrize("invalid_k1", [float("nan"), float("inf")])
def test_bm25_rejects_non_finite_k1(invalid_k1: float) -> None:
    with pytest.raises(ValueError, match="k1 must be finite and non-negative"):
        BM25Retriever([], k1=invalid_k1)


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


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_reranker_rejects_non_finite_custom_scores(invalid_score: float) -> None:
    reranker = CrossEncoderReRanker(scorer=lambda _question, _passage: invalid_score)

    with pytest.raises(ValueError, match="reranker scorer must return a finite score"):
        reranker.re_rank(["a relevant passage"], "relevant")


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


@pytest.mark.parametrize(
    "invalid_range",
    [
        (0, 2),
        (2, 1),
        (3, 2),
        (1, 1.5),
        (1, float("nan")),
        (1,),
        (1, 2, 3),
        (True, 2),
        "ab",
        None,
    ],
)
def test_tfidf_embedding_backend_rejects_invalid_ngram_range(invalid_range) -> None:
    with pytest.raises(ValueError, match="ngram_range"):
        TfidfEmbeddingBackend(ngram_range=invalid_range)


def test_tfidf_embedding_backend_accepts_valid_ngram_range() -> None:
    for valid in ((1, 2), (1, 1), (2, 3), [1, 2]):
        matrix = TfidfEmbeddingBackend(ngram_range=valid).encode_documents(["alpha beta gamma"])
        assert matrix.shape[0] == 1
        assert matrix.shape[1] > 0


def test_negative_top_k_rejected_at_every_public_boundary() -> None:
    corpus = load_corpus(SAMPLE_CORPUS)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)

    for retriever in (bm25, dense):
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            retriever.retrieve("BM25", top_k=-1)

    fusion = ReciprocalRankFusion()
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        fusion.fuse([bm25.retrieve("BM25", top_k=3)], top_k=-1)

    reranker = CrossEncoderReRanker(corpus)
    candidates = bm25.retrieve("BM25", top_k=3)
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        reranker.re_rank(candidates, "BM25", top_k=-1)
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        reranker.re_rank(candidates, "", top_k=-1)

    pipeline = RAGPipeline(corpus)
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        pipeline.retrieve("BM25", top_k=-3)
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        pipeline.run("BM25", top_k=-3)

    # zero keeps its documented empty-result behavior
    assert pipeline.retrieve("BM25", top_k=0) == []


@pytest.mark.parametrize("bad_top_k", [True, False, 2.5, "3", float("nan")])
def test_non_integer_top_k_rejected_at_every_public_boundary(bad_top_k) -> None:
    """Booleans, floats, and strings must not silently truncate or crash slicing."""
    corpus = load_corpus(SAMPLE_CORPUS)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)

    for retriever in (bm25, dense):
        with pytest.raises(ValueError, match="top_k must be non-negative and an integer"):
            retriever.retrieve("BM25", top_k=bad_top_k)

    fusion = ReciprocalRankFusion()
    with pytest.raises(ValueError, match="top_k must be non-negative and an integer"):
        fusion.fuse([bm25.retrieve("BM25", top_k=3)], top_k=bad_top_k)

    reranker = CrossEncoderReRanker(corpus)
    candidates = bm25.retrieve("BM25", top_k=3)
    with pytest.raises(ValueError, match="top_k must be non-negative and an integer"):
        reranker.re_rank(candidates, "BM25", top_k=bad_top_k)

    pipeline = RAGPipeline(corpus)
    with pytest.raises(ValueError, match="top_k must be non-negative and an integer"):
        pipeline.retrieve("BM25", top_k=bad_top_k)


def test_retrieve_rejects_none_top_k() -> None:
    """Only fuse/re_rank accept None; retrievers require an explicit count."""
    corpus = load_corpus(SAMPLE_CORPUS)
    with pytest.raises(ValueError, match="top_k must be non-negative and an integer"):
        BM25Retriever(corpus).retrieve("BM25", top_k=None)


def test_optional_top_k_none_still_means_untruncated() -> None:
    corpus = load_corpus(SAMPLE_CORPUS)
    bm25 = BM25Retriever(corpus)
    candidates = bm25.retrieve("BM25", top_k=3)

    fusion = ReciprocalRankFusion()
    assert len(fusion.fuse([candidates], top_k=None)) == len(candidates)

    reranker = CrossEncoderReRanker(corpus)
    assert len(reranker.re_rank(candidates, "BM25", top_k=None)) == len(candidates)


def test_tfidf_unfittable_tokenization_and_blank_query_contracts() -> None:
    """Single-character-only text produces an empty feature space (documented, not a
    crash), unfitted queries encode to a zero-width vector, and a blank query on a
    fitted backend encodes to the zero vector at the model width."""
    from src.models import TfidfEmbeddingBackend

    emb = TfidfEmbeddingBackend()
    out = emb.encode_documents([""])
    assert out.shape == (1, 0), "an unfittable tokenization yields an empty feature space"
    assert emb.encode_query("anything").shape == (0,), "unfitted queries are zero-width"

    fitted = TfidfEmbeddingBackend()
    fitted.encode_documents(["good text", "more text here"])
    blank = fitted.encode_query("")
    assert blank.shape[0] == fitted.encode_query("text").shape[0]
    assert float(abs(blank).sum()) == 0.0, "a blank query encodes to the zero vector"

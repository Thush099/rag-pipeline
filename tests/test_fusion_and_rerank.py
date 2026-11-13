"""Unit tests for ReciprocalRankFusion and CrossEncoderReRanker."""

import math
import pytest

from src.models import CrossEncoderReRanker, ReciprocalRankFusion
from src.utils import RetrievalResult


def test_rrf_boosts_dual_retrieved_document_above_single_list_top_hit() -> None:
    bm25_list = [
        RetrievalResult(doc_id="only-bm25", title="A", text="alpha", score=8.0, rank=1, source="bm25", bm25_score=8.0),
        RetrievalResult(doc_id="dual-doc", title="B", text="beta", score=6.0, rank=2, source="bm25", bm25_score=6.0),
    ]
    dense_list = [
        RetrievalResult(doc_id="only-dense", title="C", text="gamma", score=0.95, rank=1, source="dense", dense_score=0.95),
        RetrievalResult(doc_id="dual-doc", title="B", text="beta", score=0.80, rank=2, source="dense", dense_score=0.80),
    ]

    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse([bm25_list, dense_list])

    assert fused[0].doc_id == "dual-doc"
    assert fused[0].source == "hybrid"
    assert fused[0].bm25_score == 6.0
    assert fused[0].dense_score == 0.80
    assert math.isclose(fused[0].score, 2.0 / (60 + 2), rel_tol=1e-9)
    assert math.isclose(fused[1].score, 1.0 / (60 + 1), rel_tol=1e-9)
    assert fused[0].score > fused[1].score


def test_cross_encoder_reranker_rewards_phrase_proximity_and_coverage() -> None:
    corpus = [
        {"id": "exact", "title": "Exact", "text": "Reciprocal Rank Fusion merges ranked lists in hybrid RAG."},
        {
            "id": "scattered",
            "title": "Scattered",
            "text": (
                "Reciprocal properties appear in math while a rank matrix stores values "
                "and nuclear fusion happens in stars with hybrid engines."
            ),
        },
        {"id": "partial", "title": "Partial", "text": "Hybrid search uses sparse retrieval."},
    ]
    reranker = CrossEncoderReRanker(corpus)
    candidates = [
        RetrievalResult(doc_id="scattered", title="Scattered", text=corpus[1]["text"], score=0.5, rank=1),
        RetrievalResult(doc_id="partial", title="Partial", text=corpus[2]["text"], score=0.4, rank=2),
        RetrievalResult(doc_id="exact", title="Exact", text=corpus[0]["text"], score=0.3, rank=3),
    ]

    reranked = reranker.re_rank(candidates, "Reciprocal Rank Fusion hybrid")
    assert [r.doc_id for r in reranked] == ["exact", "scattered", "partial"]
    assert reranked[0].rerank_score > reranked[1].rerank_score > reranked[2].rerank_score
    assert reranked[0].rank == 1


def test_cross_encoder_reranker_idf_overlap_and_custom_callable() -> None:
    corpus = [
        {"id": "c1", "title": "Common", "text": "retrieval systems return documents quickly"},
        {"id": "c2", "title": "Common2", "text": "retrieval systems rank documents"},
        {"id": "c3", "title": "Rare", "text": "retrieval systems use hyde"},
    ]
    reranker = CrossEncoderReRanker(corpus)
    # Query has common word "retrieval" and rare word "hyde"
    score_rare = reranker.score_pair("retrieval hyde", corpus[2]["text"])
    score_common_only = reranker.score_pair("retrieval hyde", corpus[0]["text"])
    assert score_rare > score_common_only

    # Custom callable overrides default heuristic cleanly
    custom = CrossEncoderReRanker(scorer=lambda q, p: 42.0 if "special" in p else 1.0)
    reranked = custom.re_rank(["plain text", "special passage"], "query")
    assert reranked[0].text == "special passage"
    assert reranked[0].rerank_score == pytest.approx(42.0)

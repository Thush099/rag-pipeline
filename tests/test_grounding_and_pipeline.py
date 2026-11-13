"""Tests for AnswerGrounder, RAGPipeline, edge cases, and CLI execution."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from main import main as cli_main
from src.models import AnswerGrounder
from src.pipeline import RAGPipeline
from src.utils import RetrievalResult, load_corpus, split_sentences

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CORPUS = REPO_ROOT / "data" / "sample_corpus.jsonl"


def test_split_sentences_preserves_decimals_and_abbreviations() -> None:
    text = "Okapi BM25 uses k1=1.5 and b=0.75, e.g. for lexical retrieval. It works offline."
    sentences = split_sentences(text)
    assert sentences == [
        "Okapi BM25 uses k1=1.5 and b=0.75, e.g. for lexical retrieval.",
        "It works offline.",
    ]


def test_answer_grounder_citations_and_groundedness_discrimination() -> None:
    passages = [
        RetrievalResult(
            doc_id="rag-04",
            title="RRF",
            text=(
                "Reciprocal Rank Fusion sums 1 / (k + rank) across retrievers with k=60. "
                "Dual-retrieved passages receive a compounded boost in hybrid RAG."
            ),
            score=0.9,
            rank=1,
        ),
        RetrievalResult(
            doc_id="rag-02",
            title="BM25",
            text="Okapi BM25 uses k1=1.5 and b=0.75 for lexical ranking.",
            score=0.5,
            rank=2,
        ),
    ]
    grounder = AnswerGrounder()
    grounded = grounder.ground(
        "How does Reciprocal Rank Fusion boost hybrid RAG?", passages, max_sentences=2
    )

    assert "[rag-04]" in grounded.answer
    assert len(grounded.citations) >= 1
    assert grounded.citations[0].doc_id == "rag-04"
    assert grounded.groundedness_score >= 0.85

    ungrounded_score = grounder.score_groundedness(
        "Saturn rings consist primarily of frozen ammonia crystals orbiting Titan.",
        passages,
    )
    assert ungrounded_score < 0.25
    assert grounded.groundedness_score > ungrounded_score


def test_pipeline_modes_and_sample_corpus_retrieval() -> None:
    corpus = load_corpus(SAMPLE_CORPUS)
    assert 12 <= len(corpus) <= 15

    pipeline = RAGPipeline(corpus)
    question = "How does Reciprocal Rank Fusion improve hybrid RAG?"

    hybrid_out = pipeline.run(question, top_k=5, mode="hybrid")
    bm25_out = pipeline.run(question, top_k=3, mode="bm25")
    dense_out = pipeline.run(question, top_k=3, mode="dense")

    assert hybrid_out.results[0].doc_id == "rag-04"
    assert all(res.score > 0.0 for res in hybrid_out.results)
    assert bm25_out.results[0].doc_id == "rag-04"
    assert dense_out.results[0].doc_id == "rag-04"
    assert "[rag-04]" in hybrid_out.grounded.answer
    assert hybrid_out.grounded.groundedness_score > 0.8


def test_edge_cases_empty_query_empty_corpus_zero_topk() -> None:
    corpus = load_corpus(SAMPLE_CORPUS)
    pipeline = RAGPipeline(corpus)

    for blank in ("", "   ", "\n\t"):
        out = pipeline.run(blank, top_k=5)
        assert out.results == []
        assert out.grounded.citations == []
        assert out.grounded.groundedness_score == 0.0

    assert pipeline.retrieve("BM25", top_k=0) == []

    empty_pipeline = RAGPipeline([])
    empty_out = empty_pipeline.run("BM25", top_k=5)
    assert empty_out.results == []
    assert empty_out.grounded.groundedness_score == 0.0

    no_match_out = pipeline.run("zxqjv nonexistenttokenxyz", top_k=5)
    assert no_match_out.results == []
    assert no_match_out.grounded.groundedness_score == 0.0


def test_cli_smoke_run_and_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main([
        "--corpus",
        str(SAMPLE_CORPUS),
        "--question",
        "How does Reciprocal Rank Fusion improve hybrid RAG?",
        "--mode",
        "hybrid",
        "--top-k",
        "3",
        "--json",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["mode"] == "hybrid"
    assert len(payload["results"]) == 3
    assert payload["results"][0]["doc_id"] == "rag-04"
    assert "[rag-04]" in payload["grounded"]["answer"]


def test_cli_subprocess_default_execution() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--question",
            "How does Reciprocal Rank Fusion improve hybrid RAG?",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "rag-04" in proc.stdout
    assert "Groundedness Score:" in proc.stdout

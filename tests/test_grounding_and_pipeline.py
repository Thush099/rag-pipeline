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


@pytest.mark.parametrize(
    "invalid_threshold",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.5, "x", None, True],
)
def test_answer_grounder_rejects_invalid_min_sentence_score(invalid_threshold) -> None:
    """A malformed score threshold must fail at construction, not silently empty the answer."""
    with pytest.raises(ValueError, match="min_sentence_score"):
        AnswerGrounder(min_sentence_score=invalid_threshold)


def test_answer_grounder_accepts_valid_min_sentence_score() -> None:
    for threshold in (0.0, 0.05, 1.0):
        assert AnswerGrounder(min_sentence_score=threshold).min_sentence_score == threshold


def test_groundedness_requires_valid_citations_and_scores_the_cited_source() -> None:
    passages = [
        RetrievalResult(
            doc_id="policy-refund",
            title="Refund policy",
            text="Damaged orders delivered within 30 days qualify for a full refund.",
            score=0.9,
            rank=1,
        ),
        RetrievalResult(
            doc_id="policy-shipping",
            title="Shipping policy",
            text="Shipping delays past 7 business days qualify for shipping fee reimbursement.",
            score=0.8,
            rank=2,
        ),
    ]
    grounder = AnswerGrounder()

    correct = grounder.score_groundedness(
        "Damaged orders delivered within 30 days qualify for a full refund. [policy-refund]",
        passages,
    )
    wrong_source = grounder.score_groundedness(
        "Damaged orders delivered within 30 days qualify for a full refund. [policy-shipping]",
        passages,
    )
    unknown_source = grounder.score_groundedness(
        "Damaged orders delivered within 30 days qualify for a full refund. [missing-doc]",
        passages,
    )
    no_citation = grounder.score_groundedness(
        "Damaged orders delivered within 30 days qualify for a full refund.",
        passages,
    )

    assert correct == 1.0
    assert wrong_source < correct
    assert unknown_source == 0.0
    assert no_citation == 0.0

    boundary_citations = grounder.score_groundedness(
        "Damaged orders delivered within 30 days qualify for a full refund. [policy-refund] "
        "Shipping delays past 7 business days qualify for shipping fee reimbursement. [policy-shipping]",
        passages,
    )
    assert boundary_citations == 1.0


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


def _pipeline_result(groundedness: float):
    from src.utils import GroundedAnswer, PipelineResult

    return PipelineResult(
        question="q",
        mode="hybrid",
        results=[],
        grounded=GroundedAnswer(answer="a [rag-01]", citations=[], groundedness_score=groundedness),
    )


def test_exit_code_no_threshold_always_succeeds() -> None:
    from main import exit_code_for

    assert exit_code_for(_pipeline_result(0.0), None) == 0


def test_exit_code_gates_on_groundedness_threshold() -> None:
    from main import exit_code_for

    assert exit_code_for(_pipeline_result(0.5), 0.9) == 1
    assert exit_code_for(_pipeline_result(0.9), 0.9) == 0
    assert exit_code_for(_pipeline_result(1.0), 0.9) == 0


def test_exit_code_rejects_invalid_threshold() -> None:
    from main import exit_code_for

    for bad in (float("nan"), float("inf"), -0.1, 1.5):
        with pytest.raises(ValueError, match="min_groundedness must be a finite value"):
            exit_code_for(_pipeline_result(1.0), bad)


def test_cli_min_groundedness_gates_the_run() -> None:
    exit_ok = cli_main(["--question", "How does BM25 work?", "--min-groundedness", "0.0"])
    assert exit_ok == 0

    # An empty query yields no supported sentences, so groundedness is 0.0.
    exit_fail = cli_main(["--question", "", "--min-groundedness", "0.5"])
    assert exit_fail == 1


def test_pipeline_compare_builds_rank_matrix() -> None:
    """compare() reports per-mode dense-ordinal ranks plus a hybrid-ordered union."""
    corpus = load_corpus(Path("data/sample_corpus.jsonl"))
    pipeline = RAGPipeline(corpus=corpus)
    matrix = pipeline.compare("hybrid retrieval ranking", top_k=5)

    assert set(matrix["modes"]) == {"bm25", "dense", "hybrid"}
    for mode, rows in matrix["modes"].items():
        assert rows, f"{mode} returned no rows"
        assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1)), f"{mode} ranks must be dense ordinals"

    union = matrix["union"]
    seen = {r["doc_id"] for rows in matrix["modes"].values() for r in rows}
    assert {u["doc_id"] for u in union} == seen, "union must cover every doc seen in any mode"
    hybrid_order = [u["hybrid_rank"] for u in union if u["hybrid_rank"] is not None]
    assert hybrid_order == sorted(hybrid_order), "rows must be ordered by hybrid rank"


def test_cli_compare_prints_rank_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["--compare", "--question", "hybrid retrieval ranking", "--top-k", "5"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "bm25" in out and "dense" in out and "hybrid" in out, "table must show all three modes"
    assert "rag-05" in out, "table must list ranked documents"


def test_cli_compare_json_and_gate_conflict(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["--compare", "--json", "--question", "hybrid retrieval ranking"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["modes"]) == {"bm25", "dense", "hybrid"}
    assert payload["union"]

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--compare", "--min-groundedness", "0.5"])
    assert excinfo.value.code != 0, "--min-groundedness gates an answer; compare produces none"

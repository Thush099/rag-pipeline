"""CLI entrypoint for running the hybrid RAG pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

from config import RAGPipelineConfig
from src.pipeline import RAGPipeline
from src.utils import PipelineResult, load_corpus

DEFAULT_QUESTION = "How does Reciprocal Rank Fusion improve hybrid RAG?"
DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "sample_corpus.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the RAG pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Hybrid RAG pipeline with BM25, dense retrieval, RRF, re-ranking, and grounding."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question to answer against the corpus.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(DEFAULT_CORPUS_PATH),
        help="Path to JSONL or JSON corpus file (default: data/sample_corpus.jsonl).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "bm25", "dense"],
        default="hybrid",
        help="Retrieval mode to execute.",
    )
    parser.add_argument(
        "--top-k",
        "--top_k",
        dest="top_k",
        type=int,
        default=5,
        help="Number of top passages to return.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable CrossEncoderReRanker on candidate results.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit output as formatted JSON.",
    )
    parser.add_argument(
        "--min-groundedness",
        type=float,
        default=None,
        help=(
            "Exit non-zero when the answer's groundedness score is below this "
            "threshold in [0, 1], so the pipeline can gate a CI job."
        ),
    )
    return parser


def exit_code_for(result: PipelineResult, min_groundedness: Optional[float]) -> int:
    """Map a pipeline result to a process exit code under a groundedness gate.

    Returns 0 when no threshold is given or the answer meets it, and 1 when the
    groundedness score is below the threshold. The threshold must be a finite
    value in [0, 1]; anything else raises ``ValueError`` so a typo cannot
    silently disable gating.
    """
    if min_groundedness is None:
        return 0
    if not math.isfinite(min_groundedness) or not 0.0 <= min_groundedness <= 1.0:
        raise ValueError(
            "min_groundedness must be a finite value in [0, 1]; "
            f"got {min_groundedness!r}"
        )
    return 1 if result.grounded.groundedness_score < min_groundedness else 0


def format_report(output: PipelineResult) -> str:
    """Format a PipelineResult into a clean human-readable CLI report."""
    lines = [
        f"Question: {output.question}",
        f"Mode:     {output.mode}",
        f"Hits:     {len(output.results)}",
        "",
        "Top Retrieved Passages:",
    ]

    if not output.results:
        lines.append("  (no matching passages found)")
    else:
        for res in output.results:
            title_part = f" — {res.title}" if res.title else ""
            subscores = []
            if res.bm25_score is not None:
                subscores.append(f"bm25={res.bm25_score:.4f}")
            if res.dense_score is not None:
                subscores.append(f"dense={res.dense_score:.4f}")
            if res.rrf_score is not None:
                subscores.append(f"rrf={res.rrf_score:.4f}")
            if res.rerank_score is not None:
                subscores.append(f"rerank={res.rerank_score:.4f}")
            detail = f" ({', '.join(subscores)})" if subscores else ""
            lines.append(
                f"  [{res.rank}] [{res.doc_id}]{title_part} | score={res.score:.4f}{detail}"
            )
            snippet = res.text if len(res.text) <= 180 else f"{res.text[:177]}..."
            lines.append(f"      {snippet}")

    lines.extend(["", "Grounded Answer:"])
    if output.grounded.answer:
        lines.append(f"  {output.grounded.answer}")
    else:
        lines.append("  (no grounded supporting sentences found)")
    lines.append(
        f"Groundedness Score: {output.grounded.groundedness_score:.4f} "
        f"({len(output.grounded.citations)} citations)"
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the RAG pipeline CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        parser.error(f"corpus file not found: {corpus_path}")

    if args.min_groundedness is not None and (
        not math.isfinite(args.min_groundedness) or not 0.0 <= args.min_groundedness <= 1.0
    ):
        parser.error(
            f"--min-groundedness must be a finite value in [0, 1]; got {args.min_groundedness!r}"
        )

    try:
        corpus = load_corpus(corpus_path)
        config = RAGPipelineConfig(
            corpus_path=args.corpus,
            top_k=args.top_k,
            mode=args.mode,
            use_cross_encoder_re_ranker=not args.no_rerank,
        )
    except ValueError as exc:
        parser.error(str(exc))
    pipeline = RAGPipeline(corpus=corpus, config=config)
    rerank_flag = False if args.no_rerank else None
    result = pipeline.run(
        question=args.question,
        top_k=args.top_k,
        mode=args.mode,
        rerank=rerank_flag,
    )

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_report(result))
    return exit_code_for(result, args.min_groundedness)


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end hybrid RAG pipeline orchestrating retrieval, fusion, re-ranking, and grounding."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from config import RAGPipelineConfig, RetrievalMode
from src.models import (
    AnswerGrounder,
    BM25Retriever,
    CrossEncoderReRanker,
    DenseRetriever,
    EmbeddingInput,
    ReciprocalRankFusion,
    _validate_top_k,
)
from src.utils import DocumentInput, PipelineResult, RetrievalResult, normalize_corpus


class RAGPipeline:
    """Modular hybrid RAG pipeline with BM25, Dense retrieval, RRF, re-ranking, and grounding."""

    def __init__(
        self,
        corpus: Sequence[DocumentInput],
        config: Optional[RAGPipelineConfig] = None,
        embedding_backend: Optional[EmbeddingInput] = None,
        cross_encoder_scorer: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.config = config or RAGPipelineConfig()
        self.documents = normalize_corpus(corpus)
        self.bm25_retriever = BM25Retriever(
            self.documents,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        self.dense_retriever = DenseRetriever(
            self.documents,
            embedding_backend=embedding_backend,
        )
        self.rrf = ReciprocalRankFusion(k=self.config.rrf_k)
        self.cross_encoder_re_ranker = (
            CrossEncoderReRanker(self.documents, scorer=cross_encoder_scorer)
            if (self.config.use_cross_encoder_re_ranker or cross_encoder_scorer is not None)
            else None
        )
        self.answer_grounder = AnswerGrounder()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[RetrievalMode] = None,
        rerank: Optional[bool] = None,
    ) -> list[RetrievalResult]:
        """Retrieve ranked passages using bm25, dense, or hybrid (RRF + optional re-rank)."""
        k = self.config.top_k if top_k is None else top_k
        _validate_top_k(k)
        active_mode = mode or self.config.mode

        if k == 0 or not self.documents or not question or not question.strip():
            return []

        if active_mode == "bm25":
            candidates = self.bm25_retriever.retrieve(question, top_k=k)
            should_rerank = False if rerank is None else rerank
        elif active_mode == "dense":
            candidates = self.dense_retriever.retrieve(question, top_k=k)
            should_rerank = False if rerank is None else rerank
        elif active_mode == "hybrid":
            pool_size = min(
                len(self.documents),
                max(k * self.config.candidate_multiplier, 10),
            )
            bm25_hits = self.bm25_retriever.retrieve(question, top_k=pool_size)
            dense_hits = self.dense_retriever.retrieve(question, top_k=pool_size)
            candidates = self.rrf.fuse([bm25_hits, dense_hits])
            should_rerank = (
                self.config.use_cross_encoder_re_ranker
                if rerank is None
                else rerank
            )
        else:
            raise ValueError(
                f"Unsupported retrieval mode {active_mode!r}. Expected 'hybrid', 'bm25', or 'dense'."
            )

        if should_rerank and candidates and self.cross_encoder_re_ranker is not None:
            reranked = self.cross_encoder_re_ranker.re_rank(candidates, question)
            if self.cross_encoder_re_ranker.scorer is None:
                reranked = [item for item in reranked if item.score > 0.0]
            for final_rank, item in enumerate(reranked[:k], start=1):
                item.rank = final_rank
            return reranked[:k]

        return candidates[:k]

    def compare(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """Answer one question through every retriever mode and build a rank matrix.

        Returns per-mode dense-ordinal rank rows plus a union table ordered by
        hybrid rank; each union row carries every mode's rank (None outside that
        mode's top-k) so the RRF fusion is inspectable.
        """
        active_top_k = top_k if top_k is not None else self.config.top_k
        modes: dict[str, list[dict[str, Any]]] = {}
        for mode in ("bm25", "dense", "hybrid"):
            rows = [
                {
                    "rank": i,
                    "doc_id": row["doc_id"],
                    "title": row.get("title") or row["doc_id"],
                    "score": row["score"],
                }
                for i, row in enumerate(
                    self.run(question, top_k=active_top_k, mode=mode).to_dict()["results"],
                    start=1,
                )
            ]
            modes[mode] = rows

        by_mode = {mode: {row["doc_id"]: row for row in rows} for mode, rows in modes.items()}
        order: list[str] = []
        seen: set[str] = set()
        for mode in ("hybrid", "bm25", "dense"):
            for row in modes[mode]:
                if row["doc_id"] not in seen:
                    seen.add(row["doc_id"])
                    order.append(row["doc_id"])

        union: list[dict[str, Any]] = []
        for doc_id in order:
            entry: dict[str, Any] = {
                "doc_id": doc_id,
                "title": None,
                "bm25_rank": None,
                "dense_rank": None,
                "hybrid_rank": None,
            }
            for mode in ("bm25", "dense", "hybrid"):
                row = by_mode[mode].get(doc_id)
                if row is not None:
                    entry[f"{mode}_rank"] = row["rank"]
                    if entry["title"] is None:
                        entry["title"] = row["title"]
            union.append(entry)
        return {"modes": modes, "union": union}

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[RetrievalMode] = None,
        rerank: Optional[bool] = None,
    ) -> PipelineResult:
        """Run retrieval and citation-grounded answer extraction for a question."""
        active_mode = mode or self.config.mode
        results = self.retrieve(
            question=question,
            top_k=top_k,
            mode=active_mode,
            rerank=rerank,
        )
        grounded = self.answer_grounder.ground(
            question=question,
            passages=results,
            max_sentences=self.config.max_citation_sentences,
        )
        return PipelineResult(
            question=question,
            mode=active_mode,
            results=results,
            grounded=grounded,
        )

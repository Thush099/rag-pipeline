"""Configuration dataclasses and defaults for the hybrid RAG pipeline."""

from dataclasses import dataclass
from typing import Literal

RetrievalMode = Literal["hybrid", "bm25", "dense"]


@dataclass
class RAGPipelineConfig:
    """Configuration settings for retrieval, fusion, re-ranking, and grounding."""

    corpus_path: str = "data/sample_corpus.jsonl"
    top_k: int = 5
    mode: RetrievalMode = "hybrid"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    use_cross_encoder_re_ranker: bool = True
    max_citation_sentences: int = 3
    candidate_multiplier: int = 3


class Config:
    """Default application configuration."""

    RAG_PIPELINE = RAGPipelineConfig()

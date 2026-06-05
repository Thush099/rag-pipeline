from dataclasses import dataclass
from typing import List

@dataclass
class RAGPipelineConfig:
    corpus_path: str
    top_k: int
    use_cross_encoder_re_ranker: bool

class Config:
    RAG_PIPELINE = RAGPipelineConfig(
        corpus_path='path/to/corpus.json',
        top_k=5,
        use_cross_encoder_re_ranker=True
    )
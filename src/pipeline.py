from typing import List
from src.models import BM25Retriever, DenseRetriever, CrossEncoderReRanker
from src.utils import get_candidate_answers

class RAGPipeline:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.bm25_retriever = BM25Retriever(corpus)
        self.dense_retriever = DenseRetriever(corpus)
        self.cross_encoder_re_ranker = CrossEncoderReRanker(corpus)

    def run(self, question: str, top_k: int) -> List[dict]:
        # BM25 retrieval
        bm25_answers = self.bm25_retriever.retrieve(question, top_k)

        # Dense retrieval
        dense_answers = self.dense_retriever.retrieve(question, top_k)

        # Combine answers
        candidate_answers = get_candidate_answers(bm25_answers, dense_answers)

        # Re-ranking (optional)
        if self.cross_encoder_re_ranker is not None:
            candidate_answers = self.cross_encoder_re_ranker.re_rank(candidate_answers, question)

        # Return top-k answers
        return candidate_answers[:top_k]
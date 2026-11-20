"""Retrievers, Reciprocal Rank Fusion, Cross-Encoder Re-Ranker, and Answer Grounder."""

from __future__ import annotations

from collections import Counter
import math
from numbers import Integral
import re
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import (
    Document,
    DocumentInput,
    GroundedAnswer,
    GroundedSentence,
    RetrievalResult,
    compute_idf_map,
    extract_content_tokens,
    make_bigrams,
    normalize_corpus,
    split_sentences,
    tokenize,
)

def _validate_top_k(top_k: Optional[int], *, allow_none: bool = False) -> None:
    """Reject booleans, floats, and strings so slicing cannot silently misread top_k."""
    if top_k is None and allow_none:
        return
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0:
        raise ValueError(f"top_k must be non-negative and an integer, got {top_k!r}")


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize 1D or 2D numpy arrays safely without NaN on zero vectors."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0.0 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return arr / safe_norms


class BM25Retriever:
    """Okapi BM25 sparse lexical retriever with smoothed IDF and length normalization."""

    def __init__(
        self,
        corpus: Sequence[DocumentInput],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        if not math.isfinite(k1) or k1 < 0:
            raise ValueError("BM25 parameter k1 must be finite and non-negative.")
        if not 0.0 <= b <= 1.0:
            raise ValueError("BM25 parameter b must lie in [0.0, 1.0].")

        self.documents: List[Document] = normalize_corpus(corpus)
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or tokenize

        self.n_docs = len(self.documents)
        self.term_freqs: List[Counter[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Counter[str] = Counter()

        for doc in self.documents:
            tokens = self.tokenizer(doc.full_text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            self.doc_freqs.update(tf.keys())

        self.avgdl = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0.0
        )
        self.idf_scores: Dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }

    def idf(self, term: str) -> float:
        """Return the smoothed IDF score for a term (0.0 if absent from corpus)."""
        normalized = self.tokenizer(term)
        key = normalized[0] if normalized else term.lower()
        return self.idf_scores.get(key, 0.0)

    def score_document(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        """Compute Okapi BM25 score for a single corpus document index."""
        if not query_tokens or self.n_docs == 0 or self.avgdl == 0.0:
            return 0.0

        doc_len = self.doc_lengths[doc_idx]
        if doc_len == 0:
            return 0.0

        tf_map = self.term_freqs[doc_idx]
        norm_factor = 1.0 - self.b + self.b * (doc_len / self.avgdl)
        score = 0.0

        for term in query_tokens:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf_scores.get(term, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * norm_factor
            score += idf * (numerator / denominator)

        return float(score)

    def get_scores(self, question: str) -> List[float]:
        """Compute BM25 scores for every document in the corpus."""
        query_tokens = self.tokenizer(question)
        if not query_tokens or self.n_docs == 0:
            return [0.0] * self.n_docs
        return [self.score_document(query_tokens, i) for i in range(self.n_docs)]

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Return top-k documents ranked by descending BM25 score."""
        _validate_top_k(top_k)
        if top_k == 0 or self.n_docs == 0:
            return []

        query_tokens = self.tokenizer(question)
        if not query_tokens:
            return []

        scores = self.get_scores(question)
        ranked_indices = [
            idx for idx in sorted(
                range(self.n_docs),
                key=lambda i: (-scores[i], self.documents[i].doc_id),
            )
            if scores[idx] > 0.0
        ][:top_k]

        return [
            RetrievalResult(
                doc_id=self.documents[idx].doc_id,
                title=self.documents[idx].title,
                text=self.documents[idx].text,
                score=scores[idx],
                rank=rank,
                source="bm25",
                bm25_score=scores[idx],
            )
            for rank, idx in enumerate(ranked_indices, start=1)
        ]


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for pluggable text embedding backends."""

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode corpus texts into a 2D array of shape (n_docs, embedding_dim)."""
        ...

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query into a 1D embedding vector."""
        ...


class TfidfEmbeddingBackend:
    """Default deterministic L2-normalized TF-IDF vector embedding backend."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        if (
            not isinstance(ngram_range, (tuple, list))
            or len(ngram_range) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in ngram_range)
            or not 1 <= ngram_range[0] <= ngram_range[1]
        ):
            raise ValueError(
                "ngram_range must be a pair of integers (min_n, max_n) with 1 <= min_n <= max_n."
            )
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=tuple(ngram_range),
            sublinear_tf=True,
            norm="l2",
            token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
        )
        self._fitted = False

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Fit TF-IDF vocabulary on corpus documents and return dense L2-normalized matrix."""
        if not texts:
            self._fitted = False
            return np.zeros((0, 0), dtype=np.float32)

        try:
            matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            self._fitted = False
            return np.zeros((len(texts), 0), dtype=np.float32)

        self._fitted = True
        return _l2_normalize(matrix.toarray())

    def encode_query(self, query: str) -> np.ndarray:
        """Transform a query using the fitted TF-IDF vectorizer."""
        if not self._fitted or not query.strip():
            dim = len(self.vectorizer.vocabulary_) if self._fitted else 0
            return np.zeros(dim, dtype=np.float32)
        vec = self.vectorizer.transform([query]).toarray()[0]
        return _l2_normalize(vec)


class _CallableEmbeddingAdapter:
    """Adapts a simple batch encoding callable into an EmbeddingBackend."""

    def __init__(self, encode_fn: Callable[[Sequence[str]], np.ndarray]) -> None:
        self.encode_fn = encode_fn

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(self.encode_fn(list(texts)), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        arr = np.asarray(self.encode_fn([query]), dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] != 1:
            raise ValueError("Embedding backend must return one query embedding.")
        return arr[0] if arr.ndim == 2 else arr


EmbeddingInput = Union[EmbeddingBackend, Callable[[Sequence[str]], np.ndarray]]


class DenseRetriever:
    """Dense vector cosine-similarity retriever with a pluggable embedding backend."""

    def __init__(
        self,
        corpus: Sequence[DocumentInput],
        embedding_backend: Optional[EmbeddingInput] = None,
    ) -> None:
        self.documents: List[Document] = normalize_corpus(corpus)
        self.n_docs = len(self.documents)
        self._uses_default_tfidf = embedding_backend is None

        if embedding_backend is None:
            self.backend: EmbeddingBackend = TfidfEmbeddingBackend()
        elif isinstance(embedding_backend, EmbeddingBackend):
            self.backend = embedding_backend
        elif callable(embedding_backend):
            self.backend = _CallableEmbeddingAdapter(embedding_backend)
        else:
            raise TypeError(
                "embedding_backend must implement EmbeddingBackend or be callable."
            )

        texts = [doc.full_text for doc in self.documents]
        raw_embeddings = np.asarray(self.backend.encode_documents(texts), dtype=np.float32)
        if raw_embeddings.ndim != 2 or raw_embeddings.shape[0] != self.n_docs:
            raise ValueError("Embedding backend must return one embedding per document as a 2D array.")
        if not np.isfinite(raw_embeddings).all():
            raise ValueError("Document embeddings must contain only finite values.")
        self.doc_embeddings = _l2_normalize(raw_embeddings)

    def get_scores(self, question: str) -> List[float]:
        """Compute cosine similarity between the query and all indexed documents."""
        if self.n_docs == 0 or not question or not question.strip():
            return [0.0] * self.n_docs
        if self.doc_embeddings.ndim != 2 or self.doc_embeddings.shape[1] == 0:
            return [0.0] * self.n_docs

        query_vec = np.asarray(self.backend.encode_query(question), dtype=np.float32)
        if query_vec.ndim == 2:
            if query_vec.shape[0] != 1:
                raise ValueError("Embedding backend must return one query embedding.")
            query_vec = query_vec[0]
        if query_vec.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector.")
        if query_vec.shape[0] != self.doc_embeddings.shape[1]:
            raise ValueError("Query and document embeddings must have the same dimension.")
        if not np.isfinite(query_vec).all():
            raise ValueError("Query embedding must contain only finite values.")

        query_vec = _l2_normalize(query_vec)
        if query_vec.size == 0 or float(np.linalg.norm(query_vec)) == 0.0:
            return [0.0] * self.n_docs

        scores = self.doc_embeddings @ query_vec
        return [float(s) for s in scores]

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Return top-k documents ranked by descending cosine similarity."""
        _validate_top_k(top_k)
        if top_k == 0 or self.n_docs == 0 or not question or not question.strip():
            return []

        scores = self.get_scores(question)
        ordered = sorted(
            range(self.n_docs),
            key=lambda i: (-scores[i], self.documents[i].doc_id),
        )
        if self._uses_default_tfidf:
            ordered = [idx for idx in ordered if scores[idx] > 0.0]

        ranked_indices = ordered[:top_k]
        return [
            RetrievalResult(
                doc_id=self.documents[idx].doc_id,
                title=self.documents[idx].title,
                text=self.documents[idx].text,
                score=scores[idx],
                rank=rank,
                source="dense",
                dense_score=scores[idx],
            )
            for rank, idx in enumerate(ranked_indices, start=1)
        ]


class ReciprocalRankFusion:
    """Fuses multiple ranked result lists using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k: int = 60) -> None:
        if not math.isfinite(k) or k < 0:
            raise ValueError("RRF parameter k must be finite and non-negative.")
        self.k = k

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[RetrievalResult]],
        weights: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Merge ranked lists via sum(w / (k + rank)) and return fused results."""
        if not ranked_lists:
            return []
        if weights is not None and len(weights) != len(ranked_lists):
            raise ValueError("weights length must match ranked_lists length.")

        list_weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in list_weights):
            raise ValueError("RRF weights must be finite and non-negative.")

        rrf_scores: Dict[str, float] = {}
        best_rank: Dict[str, int] = {}
        merged: Dict[str, RetrievalResult] = {}
        sources_seen: Dict[str, List[str]] = {}

        for list_idx, result_list in enumerate(ranked_lists):
            weight = list_weights[list_idx]
            for pos, item in enumerate(result_list, start=1):
                rank = item.rank if item.rank > 0 else pos
                doc_id = item.doc_id
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight * (
                    1.0 / (self.k + rank)
                )
                best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)

                if doc_id not in merged:
                    merged[doc_id] = RetrievalResult(
                        doc_id=item.doc_id,
                        title=item.title,
                        text=item.text,
                        score=0.0,
                        bm25_score=item.bm25_score,
                        dense_score=item.dense_score,
                    )
                    sources_seen[doc_id] = []
                else:
                    existing = merged[doc_id]
                    if existing.bm25_score is None and item.bm25_score is not None:
                        existing.bm25_score = item.bm25_score
                    if existing.dense_score is None and item.dense_score is not None:
                        existing.dense_score = item.dense_score

                if item.source and item.source not in sources_seen[doc_id]:
                    sources_seen[doc_id].append(item.source)

        ordered_ids = sorted(
            merged.keys(),
            key=lambda did: (-rrf_scores[did], best_rank[did], did),
        )
        if top_k is not None:
            _validate_top_k(top_k)
            if top_k == 0:
                return []
            ordered_ids = ordered_ids[:top_k]

        fused: List[RetrievalResult] = []
        for final_rank, doc_id in enumerate(ordered_ids, start=1):
            entry = merged[doc_id]
            entry.score = float(rrf_scores[doc_id])
            entry.rrf_score = float(rrf_scores[doc_id])
            entry.rank = final_rank
            srcs = sources_seen[doc_id]
            entry.source = "hybrid" if len(srcs) > 1 else (srcs[0] if srcs else "rrf")
            fused.append(entry)
        return fused


def _min_covering_span_score(
    passage_tokens: Sequence[str],
    matched_terms: set[str],
) -> float:
    """Measure how tightly clustered matched query terms appear inside a passage."""
    if len(matched_terms) < 2:
        return 1.0 if len(matched_terms) == 1 else 0.0

    positions = [
        (idx, tok)
        for idx, tok in enumerate(passage_tokens)
        if tok in matched_terms
    ]
    if len(positions) < len(matched_terms):
        return 0.0

    window_counts: Counter[str] = Counter()
    covered = 0
    left = 0
    best_span = len(passage_tokens) + 1

    for right in range(len(positions)):
        _, r_tok = positions[right]
        if window_counts[r_tok] == 0:
            covered += 1
        window_counts[r_tok] += 1

        while covered == len(matched_terms):
            span = positions[right][0] - positions[left][0] + 1
            if span < best_span:
                best_span = span
            _, l_tok = positions[left]
            window_counts[l_tok] -= 1
            if window_counts[l_tok] == 0:
                covered -= 1
            left += 1

    if best_span > len(passage_tokens):
        return 0.0
    return min(1.0, len(matched_terms) / best_span)


class CrossEncoderReRanker:
    """Re-ranks candidate passages via query coverage, IDF overlap, and proximity."""

    def __init__(
        self,
        corpus: Optional[Sequence[DocumentInput]] = None,
        scorer: Optional[Callable[[str, str], float]] = None,
        coverage_weight: float = 0.40,
        idf_weight: float = 0.35,
        proximity_weight: float = 0.25,
    ) -> None:
        self.scorer = scorer
        self.coverage_weight = coverage_weight
        self.idf_weight = idf_weight
        self.proximity_weight = proximity_weight

        if corpus:
            docs = normalize_corpus(corpus)
            self._idf_map = compute_idf_map(doc.full_text for doc in docs)
        else:
            self._idf_map = {}

    def score_pair(
        self,
        question: str,
        passage: str,
        idf_override: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute cross-encoder relevance score for a (question, passage) pair."""
        if self.scorer is not None:
            score = float(self.scorer(question, passage))
            if not math.isfinite(score):
                raise ValueError("reranker scorer must return a finite score")
            return score

        query_content = extract_content_tokens(question)
        if not query_content:
            return 0.0

        passage_tokens = tokenize(passage)
        passage_content = extract_content_tokens(passage)
        if not passage_tokens:
            return 0.0

        unique_query_terms = list(dict.fromkeys(query_content))
        passage_term_set = set(passage_tokens)
        matched_terms = [t for t in unique_query_terms if t in passage_term_set]
        if not matched_terms:
            return 0.0

        coverage = len(matched_terms) / len(unique_query_terms)

        idf_map = idf_override if idf_override is not None else self._idf_map
        default_idf = max(idf_map.values(), default=1.0)
        total_idf = sum(idf_map.get(t, default_idf) for t in unique_query_terms)
        matched_idf = sum(idf_map.get(t, default_idf) for t in matched_terms)
        idf_overlap = matched_idf / total_idf if total_idf > 0.0 else coverage

        if len(query_content) == 1:
            proximity = 1.0
        else:
            raw_query = " ".join(tokenize(question))
            raw_passage = " ".join(passage_tokens)
            content_query = " ".join(query_content)
            content_passage = " ".join(passage_content)
            exact_phrase = (
                1.0
                if (raw_query in raw_passage or content_query in content_passage)
                else 0.0
            )

            query_bigrams = make_bigrams(query_content)
            passage_bigram_set = set(make_bigrams(passage_content)) | set(
                make_bigrams(passage_tokens)
            )
            bigram_overlap = (
                sum(1 for bg in query_bigrams if bg in passage_bigram_set)
                / len(query_bigrams)
                if query_bigrams
                else 0.0
            )
            span_score = _min_covering_span_score(passage_tokens, set(matched_terms))
            proximity = (
                0.35 * exact_phrase + 0.40 * bigram_overlap + 0.25 * span_score
            )

        total = (
            self.coverage_weight * coverage
            + self.idf_weight * idf_overlap
            + self.proximity_weight * proximity
        )
        return float(total)

    def re_rank(
        self,
        candidates: Sequence[Union[RetrievalResult, str]],
        question: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Re-rank candidate passages for the given question."""
        if top_k is not None:
            _validate_top_k(top_k)
        if not candidates:
            return []

        normalized_candidates: List[RetrievalResult] = []
        for idx, item in enumerate(candidates):
            if isinstance(item, RetrievalResult):
                normalized_candidates.append(
                    RetrievalResult(
                        doc_id=item.doc_id,
                        title=item.title,
                        text=item.text,
                        score=item.score,
                        rank=item.rank,
                        source=item.source,
                        bm25_score=item.bm25_score,
                        dense_score=item.dense_score,
                        rrf_score=item.rrf_score,
                        rerank_score=item.rerank_score,
                    )
                )
            elif isinstance(item, str):
                normalized_candidates.append(
                    RetrievalResult(
                        doc_id=f"cand-{idx}",
                        title="",
                        text=item,
                        score=0.0,
                        rank=idx + 1,
                        source="candidate",
                    )
                )
            else:
                raise TypeError(f"Unsupported candidate type: {type(item)!r}")

        if not question or not question.strip():
            result = normalized_candidates[:top_k] if top_k is not None else normalized_candidates
            return result

        idf_map = (
            self._idf_map
            if self._idf_map
            else compute_idf_map(cand.full_text for cand in normalized_candidates)
        )

        scored: List[tuple[float, int, RetrievalResult]] = []
        for orig_idx, cand in enumerate(normalized_candidates):
            rerank_score = self.score_pair(
                question, cand.full_text, idf_override=idf_map
            )
            cand.rerank_score = rerank_score
            cand.score = rerank_score
            scored.append((rerank_score, -orig_idx, cand))

        scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        if top_k is not None:
            if top_k == 0:
                return []
            scored = scored[:top_k]

        reranked: List[RetrievalResult] = []
        for rank, (_, _, cand) in enumerate(scored, start=1):
            cand.rank = rank
            reranked.append(cand)
        return reranked


class AnswerGrounder:
    """Extracts citation-backed supporting sentences and computes groundedness."""

    def __init__(self, min_sentence_score: float = 0.05) -> None:
        if (
            isinstance(min_sentence_score, bool)
            or not isinstance(min_sentence_score, (int, float))
            or not math.isfinite(min_sentence_score)
            or not 0.0 <= min_sentence_score <= 1.0
        ):
            raise ValueError(
                "min_sentence_score must be a finite number between 0.0 and 1.0."
            )
        self.min_sentence_score = float(min_sentence_score)

    def _score_sentence(
        self,
        query_terms: Sequence[str],
        sentence: str,
        idf_map: Dict[str, float],
        passage_boost: float,
    ) -> float:
        sent_tokens = tokenize(sentence)
        sent_content = extract_content_tokens(sentence)
        if not sent_tokens:
            return 0.0

        sent_set = set(sent_tokens)
        unique_q = list(dict.fromkeys(query_terms))
        matched = [t for t in unique_q if t in sent_set]
        if not matched:
            return 0.0

        coverage = len(matched) / len(unique_q)
        default_idf = max(idf_map.values(), default=1.0)
        total_idf = sum(idf_map.get(t, default_idf) for t in unique_q)
        matched_idf = sum(idf_map.get(t, default_idf) for t in matched)
        idf_overlap = matched_idf / total_idf if total_idf > 0.0 else coverage

        q_bigrams = make_bigrams(query_terms)
        s_bigram_set = set(make_bigrams(sent_content)) | set(make_bigrams(sent_tokens))
        bigram_overlap = (
            sum(1 for bg in q_bigrams if bg in s_bigram_set) / len(q_bigrams)
            if q_bigrams
            else 0.0
        )

        lexical_score = 0.45 * coverage + 0.35 * idf_overlap + 0.20 * bigram_overlap
        return float(0.80 * lexical_score + 0.20 * passage_boost)

    def ground(
        self,
        question: str,
        passages: Sequence[RetrievalResult],
        max_sentences: int = 3,
    ) -> GroundedAnswer:
        """Select top supporting sentences with [doc_id] citations and score groundedness."""
        if not passages or not question or not question.strip() or max_sentences <= 0:
            return GroundedAnswer(answer="", citations=[], groundedness_score=0.0)

        query_terms = extract_content_tokens(question)
        if not query_terms:
            return GroundedAnswer(answer="", citations=[], groundedness_score=0.0)

        idf_map = compute_idf_map(p.full_text for p in passages)
        max_passage_score = max((max(p.score, 0.0) for p in passages), default=0.0)

        candidates: List[GroundedSentence] = []
        seen_sentences: set[str] = set()

        for rank_idx, passage in enumerate(passages):
            passage_boost = (
                max(passage.score, 0.0) / max_passage_score
                if max_passage_score > 0.0
                else 1.0 / (rank_idx + 1)
            )
            for sentence in split_sentences(passage.text):
                norm_key = " ".join(tokenize(sentence))
                if not norm_key or norm_key in seen_sentences:
                    continue
                score = self._score_sentence(
                    query_terms=query_terms,
                    sentence=sentence,
                    idf_map=idf_map,
                    passage_boost=passage_boost,
                )
                if score >= self.min_sentence_score:
                    seen_sentences.add(norm_key)
                    candidates.append(
                        GroundedSentence(
                            sentence=sentence,
                            doc_id=passage.doc_id,
                            title=passage.title,
                            score=score,
                        )
                    )

        if not candidates:
            return GroundedAnswer(answer="", citations=[], groundedness_score=0.0)

        candidates.sort(key=lambda c: (-c.score, c.doc_id, c.sentence))
        selected = candidates[:max_sentences]
        cited_answer = " ".join(
            f"{item.sentence} [{item.doc_id}]" for item in selected
        )
        groundedness = self.score_groundedness(cited_answer, passages)
        return GroundedAnswer(
            answer=cited_answer,
            citations=selected,
            groundedness_score=groundedness,
        )

    def score_groundedness(
        self,
        answer: str,
        passages: Sequence[RetrievalResult],
    ) -> float:
        """Score sentence support against the retrieved sources named by citations."""
        if not answer or not answer.strip() or not passages:
            return 0.0

        passage_by_id = {passage.doc_id: passage for passage in passages}
        passage_token_sets = {
            passage.doc_id: (
                set(extract_content_tokens(passage.full_text)),
                set(tokenize(passage.full_text)),
                set(make_bigrams(extract_content_tokens(passage.full_text)))
                | set(make_bigrams(tokenize(passage.full_text))),
            )
            for passage in passages
        }

        sentence_citations: List[tuple[str, set[str]]] = []
        pending_citations: set[str] = set()
        segments = re.split(r"(\[[^\]]+\])", answer)
        for segment in segments:
            citation_match = re.fullmatch(r"\[([^\]]+)\]", segment)
            if citation_match:
                if sentence_citations:
                    sentence_citations[-1][1].add(citation_match.group(1))
                else:
                    pending_citations.add(citation_match.group(1))
                continue

            for sentence in split_sentences(segment):
                sentence_citations.append((sentence, pending_citations))
                pending_citations = set()

        sentence_scores: List[float] = []
        for sentence, citation_ids in sentence_citations:
            cited_passages = [
                passage_token_sets[doc_id]
                for doc_id in citation_ids
                if doc_id in passage_by_id
            ]
            s_content = extract_content_tokens(sentence)
            if not s_content:
                continue
            if not cited_passages:
                sentence_scores.append(0.0)
                continue

            s_unique = set(s_content)
            s_bigrams = make_bigrams(s_content)
            best_support = 0.0
            for p_content_set, p_all_set, p_bigrams in cited_passages:
                token_cov = (
                    sum(1 for t in s_unique if t in p_content_set or t in p_all_set)
                    / len(s_unique)
                )
                if s_bigrams:
                    bg_cov = (
                        sum(1 for bg in s_bigrams if bg in p_bigrams) / len(s_bigrams)
                    )
                    support = 0.70 * token_cov + 0.30 * bg_cov
                else:
                    support = token_cov
                if support > best_support:
                    best_support = support

            sentence_scores.append(best_support)

        if not sentence_scores:
            return 0.0
        return float(min(1.0, sum(sentence_scores) / len(sentence_scores)))

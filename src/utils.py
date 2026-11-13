"""Shared data structures, tokenization, and corpus I/O helpers for rag-pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "in", "into", "is", "it", "its", "may", "might", "more", "most", "no",
    "not", "of", "on", "or", "our", "should", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom",
    "why", "will", "with", "would",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ABBREVIATION_RE = re.compile(
    r"\b(?:e\.g|i\.e|vs|etc|fig|dr|mr|mrs|ms|prof|inc|ltd|al)\.",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")


@dataclass(frozen=True)
class Document:
    """Normalized corpus document."""

    doc_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Return title combined with body text for indexing and scoring."""
        return f"{self.title}. {self.text}".strip() if self.title else self.text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document to a plain dictionary."""
        return {
            "id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


DocumentInput = Union[Document, Mapping[str, Any], str]


@dataclass
class RetrievalResult:
    """Scored document returned by a retriever, fusion step, or re-ranker."""

    doc_id: str
    title: str
    text: str
    score: float
    rank: int = 0
    source: str = ""
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def full_text(self) -> str:
        """Return title combined with body text."""
        return f"{self.title}. {self.text}".strip() if self.title else self.text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retrieval result to a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class GroundedSentence:
    """Extracted supporting sentence linked to a source document ID."""

    sentence: str
    doc_id: str
    title: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grounded sentence to a plain dictionary."""
        return asdict(self)


@dataclass
class GroundedAnswer:
    """Extractive answer with inline [doc_id] citations and groundedness score."""

    answer: str
    citations: List[GroundedSentence]
    groundedness_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grounded answer to a plain dictionary."""
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "groundedness_score": self.groundedness_score,
        }


@dataclass
class PipelineResult:
    """End-to-end output produced by RAGPipeline.run()."""

    question: str
    mode: str
    results: List[RetrievalResult]
    grounded: GroundedAnswer

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full pipeline result to a JSON-compatible dictionary."""
        return {
            "question": self.question,
            "mode": self.mode,
            "results": [result.to_dict() for result in self.results],
            "grounded": self.grounded.to_dict(),
        }


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into alphanumeric tokens."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def extract_content_tokens(text: str) -> List[str]:
    """Return non-stopword alphanumeric tokens; fall back to all tokens if empty."""
    tokens = tokenize(text)
    content = [tok for tok in tokens if tok not in STOPWORDS]
    return content if content else tokens


def make_bigrams(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    """Return ordered adjacent token pairs."""
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving decimals and common abbreviations."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []

    dot_placeholder = "\u0000"
    masked = _DECIMAL_RE.sub(dot_placeholder, cleaned)
    masked = _ABBREVIATION_RE.sub(
        lambda m: m.group(0).replace(".", dot_placeholder),
        masked,
    )
    raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", masked)
    return [
        part.replace(dot_placeholder, ".").strip()
        for part in raw_parts
        if part.replace(dot_placeholder, ".").strip()
    ]


def compute_idf_map(texts: Iterable[str]) -> Dict[str, float]:
    """Compute smoothed Robertson-Sparck Jones IDF weights across a text collection."""
    doc_freqs: Counter[str] = Counter()
    n_docs = 0
    for text in texts:
        n_docs += 1
        doc_freqs.update(set(tokenize(text)))

    if n_docs == 0:
        return {}

    return {
        term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for term, df in doc_freqs.items()
    }


def normalize_corpus(corpus: Sequence[DocumentInput]) -> List[Document]:
    """Normalize heterogeneous corpus inputs (Document, dict, or str) into Documents."""
    normalized: List[Document] = []
    for idx, item in enumerate(corpus):
        if isinstance(item, Document):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(Document(doc_id=f"doc-{idx}", title="", text=item))
        elif isinstance(item, Mapping):
            doc_id = str(item.get("id") or item.get("doc_id") or f"doc-{idx}")
            title = str(item.get("title") or "")
            text = str(item.get("text") or "")
            metadata = {
                k: v for k, v in item.items() if k not in {"id", "doc_id", "title", "text"}
            }
            normalized.append(
                Document(doc_id=doc_id, title=title, text=text, metadata=metadata)
            )
        else:
            raise TypeError(f"Unsupported corpus item type at index {idx}: {type(item)!r}")
    return normalized


def load_corpus(path: Union[str, Path]) -> List[Document]:
    """Load documents from a JSONL or JSON corpus file."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    raw_text = corpus_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if corpus_path.suffix.lower() == ".json":
        payload = json.loads(raw_text)
        records = payload.get("documents", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list of documents in {corpus_path}")
        return normalize_corpus(records)

    records = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in corpus {corpus_path} at line {line_no}: {exc}"
            ) from exc
    return normalize_corpus(records)

# RAG Pipeline

A modular, offline-ready hybrid Retrieval-Augmented Generation (RAG) pipeline combining Okapi BM25 sparse retrieval, dense vector cosine retrieval, Reciprocal Rank Fusion (RRF), cross-encoder re-ranking, and citation-grounded answer extraction.

## Features

- **Okapi BM25 (`BM25Retriever`)**: Genuine sparse lexical retriever with `k1=1.5`, `b=0.75`, smoothed Robertson-Sparck Jones IDF weighting, and document length normalization.
- **Dense Vector Retrieval (`DenseRetriever`)**: Deterministic L2-normalized TF-IDF (`ngram_range=(1, 2)`, sublinear TF) cosine similarity by default, plus a pluggable `EmbeddingBackend` protocol / callable interface so external neural embeddings (e.g., SentenceTransformers, OpenAI, custom encoders) drop in without changing pipeline code. Custom backends must return finite document vectors (one per document) and one finite query vector with matching dimensions. The embedding `ngram_range` is validated as a `(min_n, max_n)` integer pair with `1 <= min_n <= max_n`, so a malformed range is rejected instead of silently producing empty embeddings.
- **Reciprocal Rank Fusion (`ReciprocalRankFusion`)**: Scale-free rank merging (`sum(1 / (k + rank))`, default `k=60`) that boosts documents retrieved strongly by both sparse and dense branches while preserving per-retriever scores. Optional per-list weights must be finite and non-negative.
- **Cross-Encoder Re-Ranking (`CrossEncoderReRanker`)**: Jointly scores `(query, passage)` candidates using query term coverage, IDF-weighted overlap, exact phrase alignment, bigram overlap, and minimum covering token span tightness—plus optional custom cross-encoder callable support.
- **Citation Grounding (`AnswerGrounder`)**: Extracts top supporting sentences with inline `[doc_id]` citations and computes a source-verification groundedness score in `[0.0, 1.0]`. The `min_sentence_score` threshold is validated as finite and within `[0.0, 1.0]`.
- **Bundled Sample Corpus**: Includes `data/sample_corpus.jsonl` with 14 technical entries covering RAG architecture, BM25, dense retrieval, RRF, re-ranking, chunking, HNSW, grounding, and evaluation. Corpus document IDs must be unique because retrieval fusion and citations use IDs to identify sources.

## Architecture

```text
Question ──┬──► BM25Retriever (Sparse, k1=1.5, b=0.75) ──┐
           │                                              ├──► ReciprocalRankFusion (k=60)
           └──► DenseRetriever (L2 Cosine / Pluggable) ──┘              │
                                                                         ▼
Grounded Answer ◄── AnswerGrounder ([doc_id] Citations) ◄── CrossEncoderReRanker
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline with the bundled sample corpus and default question, or pass custom flags:

```bash
python main.py --question "How does Reciprocal Rank Fusion improve hybrid RAG?"
```

Supported CLI options:

- `--question`: Question to answer against the corpus (default: `"How does Reciprocal Rank Fusion improve hybrid RAG?"`)
- `--corpus`: Path to `.jsonl` or `.json` corpus file (default: `data/sample_corpus.jsonl`)
- `--mode`: Retrieval strategy (`hybrid`, `bm25`, or `dense`; default: `hybrid`)
- `--top-k` / `--top_k`: Number of top passages to return (default: `5`)
- `--no-rerank`: Disable `CrossEncoderReRanker`
- `--json`: Print structured JSON output
- `--min-groundedness FLOAT`: Exit non-zero when the answer's groundedness score is below `FLOAT`, so the pipeline can gate a CI job (e.g. `python main.py --min-groundedness 0.6`). The value must be finite and in `[0, 1]`; invalid values are rejected rather than silently disabling the gate.

### Sample CLI Output

```text
$ python main.py --question "How does Reciprocal Rank Fusion improve hybrid RAG?"
Question: How does Reciprocal Rank Fusion improve hybrid RAG?
Mode:     hybrid
Hits:     4

Top Retrieved Passages:
  [1] [rag-04] — Reciprocal Rank Fusion (RRF) for Hybrid RAG | score=0.7332 (bm25=11.7529, dense=0.3148, rrf=0.0328, rerank=0.7332)
      Reciprocal Rank Fusion (RRF) merges ranked lists from heterogeneous retrievers such as sparse BM25 and dense vector search without requiring score calibration or normalization a...
  [2] [rag-01] — Retrieval-Augmented Generation (RAG) Architecture | score=0.3870 (bm25=5.7541, dense=0.1159, rrf=0.0323, rerank=0.3870)
      Retrieval-Augmented Generation (RAG) pairs a non-parametric document retriever with a language model so answers are conditioned on up-to-date external evidence rather than param...
  [3] [rag-12] — Information Retrieval Evaluation Metrics | score=0.3182 (bm25=4.0085, dense=0.0884, rrf=0.0317, rerank=0.3182)
      Standard retrieval benchmarks measure system quality using Recall@K, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG). Recall@K tracks whether releva...
  [4] [rag-14] — Semantic Caching and Latency Optimization | score=0.1751 (bm25=1.5168, dense=0.0217, rrf=0.0308, rerank=0.1751)
      Production RAG pipelines reduce serving latency and compute cost by caching normalized query embeddings, first-stage retrieval lists, and re-ranked answers. Semantic caches matc...

Grounded Answer:
  Reciprocal Rank Fusion (RRF) merges ranked lists from heterogeneous retrievers such as sparse BM25 and dense vector search without requiring score calibration or normalization across incompatible scales. [rag-04] As a result, documents retrieved strongly by both sparse and dense methods receive a compounded boost over single-retriever hits, improving recall and robustness in hybrid RAG. [rag-04] Modern production RAG systems combine lexical retrieval, dense vector search, rank fusion, and cross-encoder re-ranking before generating or extracting final answers. [rag-01]
Groundedness Score: 1.0000 (3 citations)
```

## Project Layout

```text
rag-pipeline/
├── config.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── data/
│   └── sample_corpus.jsonl
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── pipeline.py
│   └── utils.py
└── tests/
    ├── test_fusion_and_rerank.py
    ├── test_grounding_and_pipeline.py
    └── test_retrievers.py
```

## Testing

```bash
pytest -v
```

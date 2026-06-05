# RAG Pipeline

A hybrid Retrieval-Augmented Generator (RAG) pipeline for question answering, combining the strengths of sparse and dense retrieval methods.

## Problem Statement
The task of question answering involves finding relevant information from a large corpus of text to answer a given question. Traditional approaches rely on either sparse retrieval methods (e.g., BM25) or dense retrieval methods (e.g., neural networks). However, these methods have their limitations. Sparse methods can be computationally efficient but may not capture nuanced semantic relationships, while dense methods can be more accurate but are often computationally expensive.

## Architecture
```
+---------------+       +---------------+       +---------------+
|  Question    |       |  BM25 Retrieval |       |  Dense Retrieval |
|  Encoder     | ----> |  (Sparse)      | ----> |  (Dense)        |
+---------------+       +---------------+       +---------------+
                            |               |               |
                            |               |               |
                            v               v               v
+---------------+       +---------------+       +---------------+
|  Cross-Encoder|       |  Re-Ranking    |       |  Answer Grounding|
|  Re-Ranker    | <---- |  (Optional)    | <---- |  (Optional)     |
+---------------+       +---------------+       +---------------+
```

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the pipeline, use the following command:
```bash
python main.py --question "What is the capital of France?" --corpus "path/to/corpus.json"
```
This will output the top-k answers with their corresponding scores.

## Sample Output
```
Answer 1: Paris (score: 0.95)
Answer 2: Lyon (score: 0.05)
```

## Design Decisions
The architecture is designed to be modular and flexible, allowing for easy integration of different retrieval methods and re-ranking algorithms. The pipeline is also optimized for computational efficiency, using techniques such as caching and parallel processing where possible.
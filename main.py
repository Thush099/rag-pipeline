import argparse
import logging
from typing import List
from src.pipeline import RAGPipeline
from src.utils import load_corpus

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of answers to return")
    args = parser.parse_args()

    # Load corpus
    corpus = load_corpus(args.corpus)

    # Create pipeline
    pipeline = RAGPipeline(corpus)

    # Run pipeline
    answers = pipeline.run(args.question, args.top_k)

    # Print answers
    for answer in answers:
        logging.info(f"Answer: {answer['text']} (score: {answer['score']:.4f})")

if __name__ == '__main__':
    main()
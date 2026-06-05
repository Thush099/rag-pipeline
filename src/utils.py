from typing import List
import json

def load_corpus(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [json.loads(line)['text'] for line in f]

def get_candidate_answers(bm25_answers: List[str], dense_answers: List[str]) -> List[dict]:
    # Combine answers
    candidate_answers = []
    for answer in bm25_answers + dense_answers:
        candidate_answers.append({'text': answer, 'score': 0.0})
    return candidate_answers
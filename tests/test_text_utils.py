"""Unit tests for tokenization, IDF, and corpus normalization utilities."""

import math

import pytest

from src.models import Document
from src.utils import (
    compute_idf_map,
    extract_content_tokens,
    load_corpus,
    make_bigrams,
    normalize_corpus,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self) -> None:
        assert tokenize("Hello, World! 42") == ["hello", "world", "42"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert tokenize("") == []


class TestExtractContentTokens:
    def test_stopwords_are_removed(self) -> None:
        tokens = extract_content_tokens("the cat and the hat")
        assert "the" not in tokens
        assert "cat" in tokens

    def test_all_stopwords_fall_back_to_all_tokens(self) -> None:
        tokens = extract_content_tokens("the and")
        assert tokens == ["the", "and"]


class TestMakeBigrams:
    def test_adjacent_pairs_in_order(self) -> None:
        assert make_bigrams(["a", "b", "c"]) == [("a", "b"), ("b", "c")]

    def test_short_sequences_produce_no_bigrams(self) -> None:
        assert make_bigrams([]) == []
        assert make_bigrams(["solo"]) == []


class TestComputeIdfMap:
    def test_empty_corpus_returns_empty_map(self) -> None:
        assert compute_idf_map([]) == {}

    def test_rare_terms_weigh_more_than_common_terms(self) -> None:
        idf = compute_idf_map(["common rare", "common", "common"])
        assert idf["rare"] > idf["common"]

    def test_all_weights_finite_and_positive(self) -> None:
        idf = compute_idf_map(["a b", "b c", "c a"])
        assert all(math.isfinite(v) and v > 0 for v in idf.values())


class TestNormalizeCorpus:
    def test_str_items_become_documents_with_generated_ids(self) -> None:
        docs = normalize_corpus(["first", "second"])
        assert [d.doc_id for d in docs] == ["doc-0", "doc-1"]
        assert docs[0].text == "first"

    def test_dict_items_preserve_id_title_and_metadata(self) -> None:
        docs = normalize_corpus([{"id": "d1", "title": "T", "text": "body", "source": "web"}])
        doc = docs[0]
        assert doc.doc_id == "d1"
        assert doc.title == "T"
        assert doc.text == "body"
        assert doc.metadata == {"source": "web"}

    def test_document_instances_pass_through(self) -> None:
        original = Document(doc_id="keep", title="", text="x")
        assert normalize_corpus([original])[0] is original

    def test_unsupported_item_type_raises_with_index(self) -> None:
        with pytest.raises(TypeError, match="index 1"):
            normalize_corpus(["ok", 42])

    def test_duplicate_doc_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique document IDs"):
            normalize_corpus([{"id": "same", "text": "a"}, {"id": "same", "text": "b"}])


class TestLoadCorpus:
    def test_missing_file_reports_path(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="missing.jsonl"):
            load_corpus(tmp_path / "missing.jsonl")

    def test_jsonl_lines_are_loaded(self, tmp_path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"id": "a", "text": "alpha"}\n\n{"id": "b", "text": "beta"}\n')
        docs = load_corpus(path)
        assert [d.doc_id for d in docs] == ["a", "b"]

    def test_empty_file_returns_empty_corpus(self, tmp_path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_corpus(path) == []

    def test_malformed_jsonl_line_names_line_number(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "text": "alpha"}\nnot json\n')
        with pytest.raises(ValueError, match="line 2"):
            load_corpus(path)

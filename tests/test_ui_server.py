"""Tests for the RAG workbench UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_query


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_query_endpoint_runs_real_pipeline(server):
    status, payload = _post(
        server,
        {"question": "How does reciprocal rank fusion combine retrievers?", "top_k": 3},
    )
    assert status == 200
    assert payload["mode"] == "hybrid"
    assert len(payload["results"]) == 3
    assert payload["grounded"]["citations"]
    assert 0.0 <= payload["grounded"]["groundedness_score"] <= 1.0


def test_index_and_health_are_served(server):
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "Retrieval" in html and "/api/query" in html
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert health["status"] == "ok"
    assert health["modes"] == ["bm25", "dense", "hybrid"]


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"question": ""}, "question must be a non-empty string"),
        ({"question": "q", "mode": "fuzzy"}, "mode must be one of"),
        ({"question": "q", "top_k": -1}, "top_k must be non-negative"),
        ({"question": "q", "top_k": 2.5}, "top_k must be an integer"),
        ({"question": "q", "corpus_jsonl": "{oops"}, "corpus line 1 is not valid JSON"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]


def test_unknown_path_returns_404(server):
    try:
        urllib.request.urlopen(server + "/api/nope")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_run_query_with_custom_corpus():
    result = run_query(
        "What is alpha?",
        mode="bm25",
        top_k=1,
        corpus_jsonl='{"doc_id": "a1", "title": "Alpha", "text": "Alpha is a Greek letter."}',
    )
    assert result["results"][0]["doc_id"] == "a1"
    assert result["grounded"]["answer"]

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"


# --- pipeline index caching: queries must not re-embed the corpus every request ---

CORPUS_A = '{"doc_id": "a1", "title": "Alpha", "text": "Alpha is a Greek letter."}'


def test_pipeline_index_is_built_once_per_corpus(monkeypatch):
    """Repeated queries against the same corpus reuse the built index."""
    import ui.server as server_mod

    server_mod.clear_pipeline_cache()
    built = []
    real_pipeline = server_mod.RAGPipeline

    class CountingPipeline(real_pipeline):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(server_mod, "RAGPipeline", CountingPipeline)
    try:
        for _ in range(3):
            result = run_query("What is alpha?", mode="bm25", top_k=1, corpus_jsonl=CORPUS_A)
        assert result["results"][0]["doc_id"] == "a1"
        assert len(built) == 1, f"expected 1 index build for 3 queries, built {len(built)}"
    finally:
        server_mod.clear_pipeline_cache()


def test_pipeline_cache_distinguishes_corpora():
    """A cached index for one corpus never serves another corpus's answers."""
    import ui.server as server_mod

    server_mod.clear_pipeline_cache()
    try:
        run_query("What is alpha?", mode="bm25", top_k=1, corpus_jsonl=CORPUS_A)
        result = run_query(
            "What is beta?",
            mode="bm25",
            top_k=1,
            corpus_jsonl='{"doc_id": "b1", "title": "Beta", "text": "Beta is the second letter."}',
        )
        assert result["results"][0]["doc_id"] == "b1"
    finally:
        server_mod.clear_pipeline_cache()


def test_pipeline_cache_is_bounded():
    """The cache evicts oldest entries so long-running servers stay memory-bounded."""
    import ui.server as server_mod

    server_mod.clear_pipeline_cache()
    try:
        for i in range(server_mod.MAX_CACHED_PIPELINES + 2):
            run_query(
                "What is alpha?",
                mode="bm25",
                top_k=1,
                corpus_jsonl=f'{{"doc_id": "d{i}", "title": "T{i}", "text": "doc number {i} content"}}',
            )
        assert server_mod.pipeline_cache_size() <= server_mod.MAX_CACHED_PIPELINES
    finally:
        server_mod.clear_pipeline_cache()


def test_health_reports_pipeline_cache_stats(server):
    """Operational visibility: /api/health reports cache entries, hits, and misses."""
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    cache = health["pipeline_cache"]
    assert set(cache) == {"entries", "max", "hits", "misses"}
    assert cache["entries"] <= cache["max"]


def test_index_has_loading_empty_and_theme_states(server):
    """The page shows busy feedback, an empty-results state, and follows the OS theme."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'aria-busy' in html, "form must announce busy state during a run"
    assert 'class="spinner"' in html, "missing visible busy indicator"
    assert 'id="empty-state"' in html, "empty results need an explicit state"
    assert "prefers-color-scheme" in html, "page must follow the OS light/dark theme"

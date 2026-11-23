"""Lightweight local web UI server for the hybrid RAG pipeline.

Serves the self-contained workbench in ``ui/index.html`` and exposes a JSON API
that runs the real ``RAGPipeline``. Standard library only — no extra
dependencies. Start with ``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import RAGPipelineConfig, RetrievalMode  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.utils import load_corpus, normalize_corpus  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

MODES: tuple[RetrievalMode, ...] = ("bm25", "dense", "hybrid")
MAX_CORPUS_BYTES = 1_000_000


def run_query(
    question: str,
    mode: str = "hybrid",
    top_k: int = 5,
    corpus_jsonl: Optional[str] = None,
    use_cross_encoder_re_ranker: bool = True,
) -> dict[str, Any]:
    """Run one retrieval-and-grounding query and return the pipeline result.

    Raises ValueError with a clear message for invalid inputs, matching the
    CLI's boundary validation.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k must be an integer, got {top_k!r}")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    if corpus_jsonl is None:
        corpus = load_corpus(REPO_ROOT / "data" / "sample_corpus.jsonl")
    else:
        if len(corpus_jsonl.encode("utf-8")) > MAX_CORPUS_BYTES:
            raise ValueError(f"corpus exceeds {MAX_CORPUS_BYTES} bytes")
        records = []
        for line_no, line in enumerate(corpus_jsonl.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"corpus line {line_no} is not valid JSON: {exc}") from exc
        corpus = normalize_corpus(records)

    config = RAGPipelineConfig(
        mode=mode,
        top_k=top_k,
        use_cross_encoder_re_ranker=use_cross_encoder_re_ranker,
    )
    pipeline = RAGPipeline(corpus, config=config)
    return pipeline.run(question, top_k=top_k, mode=mode).to_dict()


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # keep test output quiet
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok", "modes": list(MODES)})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/query":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_query(
                    question=payload.get("question", ""),
                    mode=payload.get("mode", "hybrid"),
                    top_k=payload.get("top_k", 5),
                    corpus_jsonl=payload.get("corpus_jsonl"),
                    use_cross_encoder_re_ranker=payload.get(
                        "use_cross_encoder_re_ranker", True
                    ),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the RAG workbench UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"RAG workbench running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
ArXiv HTTP API (stdlib only)
- Part B: implements 4 endpoints
    1) GET /papers                 -> list of papers (lite fields)
    2) GET /papers/{arxiv_id}      -> full paper details
    3) GET /search?q={query}       -> search title+abstract
    4) GET /stats                  -> corpus statistics
- Part C: implementation requirements
    - C.1 Command line port: `python arxiv_server.py [port]`, default 8080
    - C.2 Data loading at startup; handle missing/bad files gracefully
    - C.3 Search: case-insensitive, term-frequency score, multi-word AND
    - C.4 Logging: print one line per request using the exact format
"""

# ---------- Standard-library imports (NO third-party deps) ----------
from http.server import HTTPServer, BaseHTTPRequestHandler   # tiny built-in HTTP server
from urllib.parse import urlparse, parse_qs                  # path + query parsing
import json                                                  # read/write JSON files
import os                                                    # cross-platform filesystem paths
import re                                                    # simple tokenization + matching
import sys                                                   # read command-line port
from datetime import datetime                                # timestamps for logs


# ============================== Paths & files (Part C.2) ==============================

# Absolute directory of this script (so relative paths work no matter where we run it)
ROOT_DIR = os.path.dirname(__file__)

# Your assignment uses `sample_data/` (keep this; change to "data" if your repo uses that)
DATA_DIR = os.path.join(ROOT_DIR, "sample_data")

# Expected files produced by the previous homework step
PAPERS_PATH = os.path.join(DATA_DIR, "papers.json")            # list[dict]
CORPUS_PATH = os.path.join(DATA_DIR, "corpus_analysis.json")   # dict (optional)


# ============================== Logging helper (Part C.4) ==============================

def log_line(method: str, path: str, status: int, note: str = "") -> None:
    """
    Print one log line using the exact format shown in the spec, e.g.:
    [2025-09-16 14:30:22] GET /papers – 200 OK (15 results)
    """
    # Minimal map of status -> reason phrase (only codes we actually emit)
    reasons = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {method} {path} – {status} {reasons.get(status, '')}"
    if note:
        msg += f" {note}"
    print(msg, flush=True)  # flush=True so CI graders see it immediately


# ============================== Robust data loading (Part C.2) ==============================

def load_json_or_none(path: str):
    """
    Try to load JSON from disk.
    - Return the parsed object on success.
    - Return None if the file is missing or malformed.
      (Server must still start; endpoints will degrade gracefully.)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# Read files once at startup (keeps requests fast and code simple)
_raw_papers = load_json_or_none(PAPERS_PATH)
# If papers file is missing or bad, fall back to an empty list
PAPERS = _raw_papers if isinstance(_raw_papers, list) else []

# O(1) lookup for GET /papers/{arxiv_id}
PAPERS_BY_ID = {p.get("arxiv_id"): p for p in PAPERS if isinstance(p, dict) and p.get("arxiv_id")}

# “Lite” list for GET /papers (only the fields required by the handout)
PAPERS_LITE = [
    {
        "arxiv_id": p.get("arxiv_id"),
        "title": p.get("title"),
        "authors": p.get("authors", []),
        "categories": p.get("categories", []),
    }
    for p in PAPERS
    if isinstance(p, dict)
]

# Precomputed stats are optional; if missing we will compute from PAPERS on demand
CORPUS_STATS = load_json_or_none(CORPUS_PATH)


# ============================== Search utilities (Part C.3) ==============================

WORD_RE = re.compile(r"\w+")  # letters/digits/underscore; good enough for this task

def tokenize(text: str) -> list[str]:
    """Lowercase and split to word tokens (makes search case-insensitive)."""
    return WORD_RE.findall((text or "").lower())

def compute_corpus_stats() -> dict:
    """
    Build the /stats data from `PAPERS` when corpus_analysis.json is not present.
    Returns a dict with exactly the keys shown in the spec.
    """
    tokens: list[str] = []
    category_counts: dict[str, int] = {}

    for p in PAPERS:
        tokens += tokenize(p.get("abstract", ""))
        for c in p.get("categories", []) or []:
            category_counts[c] = category_counts.get(c, 0) + 1

    # Frequency table for top-10 words
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    top10 = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "total_papers": len(PAPERS),
        "total_words": len(tokens),
        "unique_words": len(set(tokens)),
        "top_10_words": [{"word": w, "frequency": n} for w, n in top10],
        "category_distribution": category_counts,
    }

def search_papers_multi_and(query: str) -> list[dict]:
    """
    Implement the assignment search rules (Part C.3):
      - Case-insensitive search over title and abstract
      - Multi-word queries: **AND** semantics (all terms must appear at least once)
      - Score = total term frequency across title+abstract (higher is better)
      - Return at most 50 hits sorted by score desc
    """
    terms = [t for t in tokenize(query) if t]
    if not terms:
        # Treat as a malformed search query (caller will return 400)
        raise ValueError("empty query")

    results: list[dict] = []

    for p in PAPERS:
        title = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()

        # Count occurrences of each term across both fields
        per_term_hits = []
        for t in terms:
            hits = title.count(t) + abstract.count(t)  # simple substring count is enough here
            per_term_hits.append(hits)

        # AND semantics: every term must appear at least once
        if not all(h > 0 for h in per_term_hits):
            continue

        # Build result row as shown in the handout
        has_title_match = any(title.count(t) > 0 for t in terms)
        has_abs_match = any(abstract.count(t) > 0 for t in terms)
        results.append({
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "match_score": int(sum(per_term_hits)),
            "matches_in": [k for k, ok in (("title", has_title_match), ("abstract", has_abs_match)) if ok],
        })

    # Highest score first, cap output size
    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results[:50]


# ============================== HTTP helpers ==============================

def send_json(h: BaseHTTPRequestHandler, obj, status: int = 200) -> None:
    """Serialize `obj` as UTF-8 JSON and send with the given HTTP status."""
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


# ============================== HTTP handler (Parts B & C) ==============================

class Handler(BaseHTTPRequestHandler):
    """Routes only GET requests; suppresses default noisy logging."""

    # Silence BaseHTTPRequestHandler’s default console logs (we print our own)
    def log_message(self, *args, **kwargs):  # noqa: D401
        return

    def do_GET(self) -> None:
        """Dispatch GET requests to the 4 required endpoints with proper error handling."""
        parsed = urlparse(self.path)                         # e.g., path="/search", query="q=ml"
        parts = [seg for seg in parsed.path.split("/") if seg]  # ["papers"], ["papers","2301.12345"], ...

        try:
            # ---------- Part B.1: GET /papers ----------
            if parts == ["papers"]:
                send_json(self, PAPERS_LITE, 200)
                log_line("GET", parsed.path, 200, f"({len(PAPERS_LITE)} results)")
                return

            # ---------- Part B.2: GET /papers/{arxiv_id} ----------
            if len(parts) == 2 and parts[0] == "papers":
                arxiv_id = parts[1]
                paper = PAPERS_BY_ID.get(arxiv_id)
                if paper is None:
                    send_json(self, {"error": "not_found", "message": "unknown paper id"}, 404)
                    log_line("GET", parsed.path, 404)
                else:
                    send_json(self, paper, 200)
                    log_line("GET", parsed.path, 200)
                return

            # ---------- Part B.3: GET /search?q={query} ----------
            if parts == ["search"]:
                qs = parse_qs(parsed.query)
                q = qs.get("q", [""])[0]  # first value; may be an empty string
                if not isinstance(q, str) or q.strip() == "":
                    # Part C.3 + error handling: malformed query -> 400
                    send_json(self, {"error": "bad_request", "message": "missing or empty 'q' parameter"}, 400)
                    log_line("GET", self.path, 400)
                    return
                try:
                    results = search_papers_multi_and(q)  # may raise ValueError("empty query")
                except ValueError as e:
                    send_json(self, {"error": "bad_request", "message": str(e)}, 400)
                    log_line("GET", self.path, 400)
                    return

                send_json(self, {"query": q, "results": results}, 200)
                log_line("GET", self.path, 200, f"({len(results)} results)")
                return

            # ---------- Part B.4: GET /stats ----------
            if parts == ["stats"]:
                stats = CORPUS_STATS if isinstance(CORPUS_STATS, dict) else compute_corpus_stats()
                send_json(self, stats, 200)
                log_line("GET", parsed.path, 200)
                return

            # ---------- Invalid endpoint (Part B error handling) ----------
            send_json(self, {"error": "not_found", "message": "invalid endpoint"}, 404)
            log_line("GET", parsed.path, 404)

        except Exception as e:
            # ---------- Unexpected server error -> 500 with JSON body ----------
            send_json(self, {"error": "server_error", "message": f"{type(e).__name__}: {e}"}, 500)
            log_line("GET", parsed.path, 500)


# ============================== Bootstrap (Part C.1) ==============================

def parse_port(default_port: int = 8080) -> int:
    """
    Read optional port from the command line:
        python arxiv_server.py [port]
    - If missing or invalid, return `default_port` (8080).
    """
    if len(sys.argv) >= 2:
        try:
            return int(sys.argv[1])
        except ValueError:
            return default_port
    return default_port

def main() -> None:
    """Create the HTTP server and start serving forever."""
    port = parse_port(8080)  # default required by the spec
    server = HTTPServer(("0.0.0.0", port), Handler)  # bind all interfaces (works in Docker & local)
    print(f"Serving on http://0.0.0.0:{port}", flush=True)  # tiny startup hint for humans/CI
    server.serve_forever()  # blocking loop

if __name__ == "__main__":
    main()

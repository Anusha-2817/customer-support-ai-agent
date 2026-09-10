"""Blind reply-scoring tool for judge validation. Standard library only.

    python tools/score_server.py                  # then open http://localhost:8766
    python tools/score_server.py --only-ids data/human_scores/rescore_ids.json \
        --out data/human_scores/scores_round2.jsonl --seed 9      # self-agreement round

Shows one candidate reply at a time with the customer's message and earlier turns, and asks for
the same five 1-3 scores and the same "sendable" decision the LLM judge gives, using the judge's
exact anchors (imported from src/judge.py), so human and judge answer the same question.

Blind: the page never receives which system wrote the reply, the sampling stratum, the dev/test
split or the case id. Each score is appended to disk as soon as it's saved (crash-safe,
resumable); editing an item appends a new record and the last one wins.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from judge import DIMENSIONS  # noqa: E402

HS = ROOT / "data" / "human_scores"
HTML = Path(__file__).resolve().parent / "score.html"
VISIBLE = ("item_id", "context", "customer_msg", "reply")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(rec: dict, ids: set[str]) -> str | None:
    if str(rec.get("item_id")) not in ids:
        return "unknown item"
    for d in DIMENSIONS:
        if rec.get(d) not in (1, 2, 3):
            return f"score '{d}' 1, 2 or 3"
    if not isinstance(rec.get("sendable"), bool):
        return "decide whether BA could post it as-is"
    return None


def make_handler(items: list[dict], out: Path):
    lock = threading.Lock()
    ids = {it["item_id"] for it in items}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body, ctype: str = "application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, HTML.read_bytes(), "text/html")
            if self.path == "/api/state":
                scores = {r["item_id"]: r for r in read_jsonl(out)}          # last wins
                return self._send(200, {"rubric": DIMENSIONS, "items": items, "scores": scores, "out": out.name})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/score":
                return self._send(404, {"error": "not found"})
            rec = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            err = validate(rec, ids)
            if err:
                return self._send(400, {"error": err})
            keep = {"item_id": rec["item_id"], **{d: rec[d] for d in DIMENSIONS}, "sendable": rec["sendable"],
                    "note": str(rec.get("note") or ""), "seconds": rec.get("seconds"),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with lock:
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(keep, ensure_ascii=False) + "\n")
            self._send(200, {"ok": True})

    return Handler


def load_items(path: Path, only_ids: Path | None, seed: int) -> list[dict]:
    items = [{k: it[k] for k in VISIBLE} for it in read_jsonl(path)]
    if only_ids:
        keep = {str(x) for x in json.loads(only_ids.read_text(encoding="utf-8"))}
        items = [it for it in items if it["item_id"] in keep]
    random.Random(seed).shuffle(items)
    return items


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=HS / "items.jsonl")
    ap.add_argument("--out", type=Path, default=HS / "scores_round1.jsonl")
    ap.add_argument("--only-ids", type=Path, help="JSON list of item_ids (self-agreement round)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    if not a.items.exists():
        sys.exit(f"{a.items} not found: run python src/human_items.py --run <predictions run> first")
    items = load_items(a.items, a.only_ids, a.seed)
    print(f"{len(items)} replies to score | scores -> {a.out}")
    print(f"open http://localhost:{a.port}   (Ctrl+C to stop; progress is saved)")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(items, a.out)).serve_forever()


if __name__ == "__main__":
    main()

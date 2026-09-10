"""Blind labelling tool for the golden set. Standard library only.

    python tools/label_server.py            # round 1, then open http://localhost:8765
    python tools/label_server.py --only-ids data/golden/relabel_ids.json \
        --out data/golden/labels_round2.jsonl --seed 8   # blind re-label round

Choices here that affect label quality:
- Blind: model output is never shown, so labels can't be anchored by the system
  being evaluated.
- Staged reveal: BA's real reply stays hidden until intent and escalation are set;
  otherwise how BA actually handled the case leaks into "should this escalate?".
- The sampling stratum (uniform vs targeted) is never sent to the browser: knowing
  an item was picked as a likely escalation would bias the label.
- Fixed seeded order, so labelling fatigue doesn't line up with any data property.
- Each label is appended to disk as soon as it's saved: crash-safe and resumable.
  Editing an earlier item appends a new record; the last record per case wins.
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
GOLD = ROOT / "data" / "golden"
HTML = Path(__file__).resolve().parent / "label.html"
REASONS = ROOT / "config" / "escalation_reasons.json"
VISIBLE = ("case_id", "context", "customer_msg", "brand_reply")
BA_OK = {"yes", "no", "unsure"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(rec: dict, taxonomy: dict) -> str | None:
    """Return an error message, or None if the label is complete."""
    intents = {i["id"] for i in taxonomy["intents"]}
    reasons = {r["id"] for r in taxonomy["escalation_reasons"]}
    if rec.get("intent") not in intents:
        return "pick an intent"
    if not isinstance(rec.get("escalate"), bool):
        return "pick auto or escalate"
    if rec["escalate"] and rec.get("reason") not in reasons:
        return "escalated items need a reason"
    if rec.get("ba_reply_ok") not in BA_OK:
        return "rate BA's actual reply"
    return None


def make_handler(items: list[dict], taxonomy: dict, out: Path, practice: bool = False):
    lock = threading.Lock()
    ids = {str(it["case_id"]) for it in items}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
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
                labels = {str(r["case_id"]): r for r in read_jsonl(out)}  # last wins
                return self._send(200, {"taxonomy": taxonomy, "items": items,
                                        "labels": labels, "out": out.name,
                                        "practice": practice})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, {"error": "not found"})
            rec = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if str(rec.get("case_id")) not in ids:
                return self._send(400, {"error": "unknown case_id"})
            err = validate(rec, taxonomy)
            if err:
                return self._send(400, {"error": err})
            if not rec["escalate"]:
                rec["reason"] = None
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with lock:
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._send(200, {"ok": True})

    return Handler


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=GOLD / "to_label.jsonl")
    ap.add_argument("--taxonomy", type=Path, default=GOLD / "taxonomy.json")
    ap.add_argument("--out", type=Path, default=GOLD / "labels_round1.jsonl")
    ap.add_argument("--only-ids", type=Path, help="JSON list of case_ids (re-label round)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--practice", action="store_true",
                    help="show a practice banner; pair with a scratch --out")
    a = ap.parse_args()

    if not a.taxonomy.exists():
        sys.exit(f"{a.taxonomy} not found - finalise the taxonomy first")
    taxonomy = json.loads(a.taxonomy.read_text(encoding="utf-8"))
    # Escalation reasons come from one shared file (also read by the agent and the
    # metrics), never from the taxonomy file, so their ids cannot drift apart.
    taxonomy["escalation_reasons"] = json.loads(REASONS.read_text(encoding="utf-8"))["reasons"]
    items = [{k: it[k] for k in VISIBLE} for it in read_jsonl(a.items)]
    if a.only_ids:
        keep = {str(x) for x in json.loads(a.only_ids.read_text(encoding="utf-8"))}
        items = [it for it in items if str(it["case_id"]) in keep]
    for it in items:                       # tweet ids exceed JS's 2^53 safe range
        it["case_id"] = str(it["case_id"])
    random.Random(a.seed).shuffle(items)

    if a.practice and a.out.resolve().parent == GOLD.resolve():
        sys.exit("--practice must not write into data/golden/ - pass a scratch --out")
    print(f"{len(items)} items | labels -> {a.out}" + ("  [PRACTICE]" if a.practice else ""))
    print(f"open http://localhost:{a.port}   (Ctrl+C to stop; progress is saved)")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(items, taxonomy, a.out, a.practice)).serve_forever()


if __name__ == "__main__":
    main()

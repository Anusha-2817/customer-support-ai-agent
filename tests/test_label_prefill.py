"""Tests for practice pre-sets in the labelling tool (decision log #20).

    python tests/test_label_prefill.py

Runs the real request handler in-process against a TEMPORARY labels file, using the real
200 golden cases and the real pre-set file read-only. The real labels file
(data/golden/labels_round1.jsonl) is never created or touched. Exit code 1 on failure.
"""
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import label_server as ls  # noqa: E402

GOLD = ROOT / "data" / "golden"
REAL_LABELS = GOLD / "labels_round1.jsonl"
existed_before = REAL_LABELS.exists()
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def request(port: int, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


taxonomy = json.loads((GOLD / "taxonomy.json").read_text(encoding="utf-8"))
taxonomy["escalation_reasons"] = json.loads(ls.REASONS.read_text(encoding="utf-8"))["reasons"]
items = [{**{k: it[k] for k in ls.VISIBLE}, "case_id": str(it["case_id"])}
         for it in ls.read_jsonl(GOLD / "to_label.jsonl")]
prefill = {str(r["case_id"]): r for r in ls.read_jsonl(GOLD / "prefill_from_practice.jsonl")}

tmp = Path(tempfile.mkdtemp(prefix="prefill_test_"))
out = tmp / "labels.jsonl"
port = free_port()
srv = ThreadingHTTPServer(("127.0.0.1", port), ls.make_handler(items, taxonomy, out, False, prefill))
threading.Thread(target=srv.serve_forever, daemon=True).start()


def label(p: dict, **over) -> dict:
    rec = {"case_id": p["case_id"], "intent": p["intent"], "escalate": p["escalate"], "reason": p["reason"],
           "ba_reply_ok": p["ba_reply_ok"], "ambiguous": False, "note": p["note"]}
    rec.update(over)
    return rec


try:
    code, st = request(port, "/api/state")
    check(code == 200 and len(st["prefill"]) == 57 and set(st["prefill"]) <= {i["case_id"] for i in items},
          "state carries 57 pre-sets, all for golden cases")
    full = next(p for p in prefill.values() if p["intent"] and not p["rejudge_escalation"])
    fresh = next(p for p in prefill.values() if p["rejudge_escalation"])
    plain = next(i for i in items if i["case_id"] not in prefill)

    code, _ = request(port, "/api/label", label(full))
    saved = ls.read_jsonl(out)[-1]
    check(code == 200 and saved["prefilled_from_practice"] and saved["prior_exposure_to_ba_reply"]
          and saved["changed_from_prefill"] == [], "accepting every pre-set records provenance and no changes")

    flipped = label(full, escalate=not full["escalate"], reason=None if full["escalate"] else "unclear_request")
    code, _ = request(port, "/api/label", flipped)
    saved = ls.read_jsonl(out)[-1]
    check(code == 200 and "escalate" in saved["changed_from_prefill"],
          f"flipping the escalation is recorded as a change: {saved['changed_from_prefill']}")

    code, _ = request(port, "/api/label", label(fresh, intent=fresh["intent"] or "other_non_actionable",
                                                escalate=True, reason="unclear_request", ba_reply_ok="yes"))
    saved = ls.read_jsonl(out)[-1]
    check(code == 200 and "escalate" not in saved["prefill_fields"] and "escalate" not in saved["changed_from_prefill"],
          "a fresh judgement on a pre-rule case is not counted as a change")

    code, _ = request(port, "/api/label", {"case_id": plain["case_id"], "intent": "baggage", "escalate": False,
                                           "reason": None, "ba_reply_ok": "yes", "ambiguous": False, "note": ""})
    saved = ls.read_jsonl(out)[-1]
    check(code == 200 and saved["prefilled_from_practice"] is False and saved["prior_exposure_to_ba_reply"] is False
          and "prefill_fields" not in saved, "an unexposed case is recorded as unexposed")

    code, _ = request(port, "/api/label", label(full, intent=None))
    check(code == 400, "a blank intent is still rejected: pre-sets can't be saved unreviewed")
finally:
    srv.shutdown()

# Pre-sets must never reach a blind re-label round.
ids_file = tmp / "ids.json"
ids_file.write_text(json.dumps(list(prefill)[:5]), encoding="utf-8")
p2 = free_port()
proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "label_server.py"), "--only-ids", str(ids_file),
                         "--out", str(tmp / "round2.jsonl"), "--port", str(p2)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    code, st = None, {}
    for _ in range(50):
        try:
            code, st = request(p2, "/api/state")
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    check(code == 200 and len(st.get("items", [])) == 5 and st.get("prefill") == {}, "no pre-sets in a re-label round")
finally:
    proc.terminate()
    proc.wait()

shutil.rmtree(tmp, ignore_errors=True)
check(REAL_LABELS.exists() == existed_before, "real labels file untouched")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

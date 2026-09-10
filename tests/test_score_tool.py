"""Tests for the blind human-scoring set and tool (src/human_items.py, tools/score_server.py).

    python tests/test_score_tool.py

Builds the scoring set from the real full-v1 predictions into a temporary directory (predictions
and case texts only; the golden labels are fenced off and any attempt to open them fails), then
runs the scoring server in-process against a temporary scores file. The real data/human_scores/
files are never created or touched. Exit code 1 on any failure.
"""
import builtins
import io
import json
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = (ROOT / "data" / "golden" / "labels_round1.jsonl").resolve()
FENCE_HITS: list[str] = []
_open = builtins.open


def fenced(file, *args, **kwargs):
    try:
        hit = Path(file).resolve() == LABELS
    except TypeError:
        hit = False
    if hit:
        FENCE_HITS.append(str(file))
        raise AssertionError("tried to open the golden labels")
    return _open(file, *args, **kwargs)


builtins.open = io.open = fenced
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import human_items as H  # noqa: E402
import score_server as SS  # noqa: E402
from judge import DIMENSIONS  # noqa: E402

RUN = ROOT / "artifacts" / "predictions" / "full-v1"
REAL_HS = ROOT / "data" / "human_scores"
real_before = sorted(p.name for p in REAL_HS.glob("*")) if REAL_HS.exists() else None
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


tmp = Path(tempfile.mkdtemp(prefix="score_tool_"))
try:
    items = H.select(RUN, tmp / "items.jsonl", tmp / "scores_round1.jsonl")
    again = H.select(RUN, tmp / "items2.jsonl", tmp / "scores_round1.jsonl")
    cases = {str(json.loads(l)["case_id"]): json.loads(l) for l in open(ROOT / "data/golden/to_label.jsonl", encoding="utf-8")}
    strat = {k: sum(it["strat"] == k for it in items) for k in H.QUOTA}
    check(len(items) == 80 and strat["A_fail"] + strat["A_pass"] == 45 and strat["B1"] == 30 and strat["B0"] == 5,
          f"80 items: 45 from A (fail {strat['A_fail']}, pass {strat['A_pass']}), 30 from B1, 5 from B0")
    check(len({it["case_id"] for it in items}) == 80 and all(cases[it["case_id"]]["stratum"] == "uniform" for it in items),
          "80 different cases, all from the uniform headline set")
    check({it["source"] for it in items} == {"A", "B1", "B0-auto-all"}, "BA's actual reply is never a candidate")
    check(sum(it["split"] == "dev" for it in items) == 30 and sum(it["split"] == "test" for it in items) == 50, "fixed 30 dev / 50 test split")
    check([it["item_id"] for it in items] == [it["item_id"] for it in again], "rebuilding gives the same items in the same order")
    check(all(it["reply"] for it in items), "every item has a reply to score")
    (tmp / "scores_round1.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        H.select(RUN, tmp / "items3.jsonl", tmp / "scores_round1.jsonl")
        check(False, "the scoring set is frozen once scoring starts")
    except SystemExit:
        check(True, "the scoring set is frozen once scoring starts")

    out = tmp / "scores_test.jsonl"
    served = SS.load_items(tmp / "items.jsonl", None, seed=3)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SS.make_handler(served, out))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    code, st = call("/api/state")
    keys = {k for it in st["items"] for k in it}
    check(code == 200 and keys == {"item_id", "context", "customer_msg", "reply"},
          f"blind: the page gets only the item id, thread, message and reply ({sorted(keys)})")
    check(st["rubric"] == {k: list(v) for k, v in DIMENSIONS.items()}, "the scorer sees the judge's exact rubric anchors")
    first = st["items"][0]["item_id"]
    good = {"item_id": first, **{d: 2 for d in DIMENSIONS}, "sendable": False, "note": "x", "seconds": 5}
    code, _ = call("/api/score", good)
    saved = [json.loads(l) for l in open(out, encoding="utf-8")]
    check(code == 200 and saved[-1]["item_id"] == first and saved[-1]["sendable"] is False, "a complete score is saved")
    for bad, why in [({**good, "relevance": 4}, "a score outside 1-3"), ({k: v for k, v in good.items() if k != "sendable"}, "no sendable decision"),
                     ({**good, "item_id": "nope"}, "an unknown item")]:
        code, _ = call("/api/score", bad)
        check(code == 400, f"rejects {why}")
    code, _ = call("/api/score", {**good, "tone": 3})
    code, st = call("/api/state")
    check(st["scores"][first]["tone"] == 3 and len(st["scores"]) == 1, "editing an item keeps the latest score")
    srv.shutdown()
    ids_file = tmp / "rescore.json"
    ids_file.write_text(json.dumps([it["item_id"] for it in items[:25]]), encoding="utf-8")
    check(len(SS.load_items(tmp / "items.jsonl", ids_file, seed=9)) == 25, "a 25-item self-agreement round can be served")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    builtins.open = io.open = _open

check(not FENCE_HITS, f"never tried to open the golden labels ({len(FENCE_HITS)} attempts)")
after = sorted(p.name for p in REAL_HS.glob("*")) if REAL_HS.exists() else None
check(after == real_before, "real data/human_scores untouched")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

"""Tests for the blind re-label round (src/relabel.py), with SYNTHETIC labels in a temp directory.

    python tests/test_relabel.py

Picking the cases must not read any labels, so the real labels files are fenced off: any attempt
to open them fails the test. The synthetic rounds only exercise the code. Exit code 1 on failure.
"""
import builtins
import io
import json
import random
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden"
FENCED = {(GOLD / "labels_round1.jsonl").resolve(), (GOLD / "labels_round2.jsonl").resolve()}
FENCE_HITS: list[str] = []
_open = builtins.open


def fenced(file, *args, **kwargs):
    try:
        hit = Path(file).resolve() in FENCED
    except TypeError:
        hit = False
    if hit:
        FENCE_HITS.append(str(file))
        raise AssertionError("tried to open a real labels file")
    return _open(file, *args, **kwargs)


builtins.open = io.open = fenced
sys.path.insert(0, str(ROOT / "src"))
import relabel as R  # noqa: E402

failures: list[str] = []
real_ids_before = R.IDS.exists()


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


tmp = Path(tempfile.mkdtemp(prefix="relabel_"))
try:
    cases = {str(c["case_id"]): c for c in R.read_jsonl(R.CASES)}
    ids = R.pick(out=tmp / "ids.json", round2=tmp / "none.jsonl")
    strata = [cases[i]["stratum"] for i in ids]
    check(len(ids) == 30 and len(set(ids)) == 30, "30 distinct cases")
    check(strata.count("uniform") == 20 and strata.count("targeted") == 10, "20 uniform and 10 targeted")
    check(all(isinstance(i, str) for i in ids), "ids are strings (tweet ids exceed JavaScript's safe integers)")
    scored = {str(r["case_id"]) for r in R.read_jsonl(R.ITEMS)}
    check(len(scored) == 80 and not set(ids) & scored, "no re-labelled case is in the blind reply-scoring set")
    check(R.pick(out=tmp / "ids2.json", round2=tmp / "none.jsonl") == ids, "the pick is stable")
    (tmp / "r2_exists.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        R.pick(out=tmp / "ids3.json", round2=tmp / "r2_exists.jsonl")
        check(False, "pick refuses once round-2 labels exist")
    except SystemExit:
        check(True, "pick refuses once round-2 labels exist")
    try:
        R.agree(r2_path=tmp / "none.jsonl", ids_path=tmp / "ids.json", out_dir=tmp)
        check(False, "agree refuses without round-2 labels")
    except SystemExit as e:
        check("Nothing is substituted" in str(e), "agree refuses without round-2 labels (nothing substituted)")

    # SYNTHETIC rounds: round 2 copies round 1 except three deliberate changes.
    rng = random.Random(0)
    intents = ["baggage", "flight_disruption", "booking_change", "travel_info"]
    t0 = datetime(2026, 9, 10, 20, 0, 0)
    r1 = {i: {"case_id": i, "intent": rng.choice(intents), "escalate": rng.random() < .5, "reason": None,
              "ba_reply_ok": rng.choice(["yes", "no"]), "ts": (t0 + timedelta(minutes=k)).isoformat()}
          for k, i in enumerate(cases)}
    for r in r1.values():
        r["reason"] = "repeated_contact" if r["escalate"] else None
    r2 = {i: {**r1[i], "ts": (datetime.fromisoformat(r1[i]["ts"]) + timedelta(hours=30)).isoformat()} for i in ids}
    r2[ids[0]]["intent"] = "other_non_actionable"
    r2[ids[1]]["escalate"] = not r2[ids[1]]["escalate"]
    r2[ids[1]]["reason"] = None if r1[ids[1]]["escalate"] else "safety_medical"
    r2[ids[2]]["ba_reply_ok"] = "unsure"
    r2[ids[3]]["ts"] = (datetime.fromisoformat(r1[ids[3]]["ts"]) + timedelta(hours=2)).isoformat()
    for name, rows in (("r1.jsonl", r1.values()), ("r2.jsonl", r2.values())):
        with open(tmp / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    res = R.agree(tmp / "r1.jsonl", tmp / "r2.jsonl", tmp / "ids.json", tmp)
    check(res["n"] == 30, "agreement over the 30 re-labelled cases")
    check(res["intent"]["agreement"] == round(29 / 30, 4) and res["escalate"]["agreement"] == round(29 / 30, 4)
          and res["ba_reply_ok"]["agreement"] == round(29 / 30, 4), "one change per field is counted")
    check(all(0 < res[f]["kappa"] < 1 for f in R.FIELDS), f"kappa per field: {[res[f]['kappa'] for f in R.FIELDS]}")
    check(len(res["changed"]) == 3, "the three changed cases are listed")
    check(res["under_min_hours"] == 1 and res["hours_between_rounds"]["min"] == 2.0,
          "a re-label under 24 hours after the first is flagged")
    check((tmp / "relabel_agreement.md").exists(), "relabel_agreement.md written")
finally:
    builtins.open = io.open = _open
    shutil.rmtree(tmp, ignore_errors=True)

check(not FENCE_HITS, f"never opened the real labels files ({len(FENCE_HITS)} attempts)")
check(R.IDS.exists() == real_ids_before, "the real relabel_ids.json was not touched")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

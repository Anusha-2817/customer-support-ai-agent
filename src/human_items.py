"""Select the replies for blind human scoring, used to validate the LLM judge (DESIGN.md section 5).

    python src/human_items.py --run artifacts/predictions/full-v1

80 items from the 130 uniform (headline) cases, one reply each:
  A  (the agent)           45 items: 15 whose draft fails the deterministic reply checks and 30
                           that pass, so the scorer sees the full range of quality;
  B1 (a past BA reply)     30 items;
  B0 (the fixed reply)      5 items, a generic-reply anchor.
Each case appears once, so there is no side-by-side comparison to anchor on. BA's actual reply
for the case is never a candidate: the labeller already saw every one of them while labelling.
Items are shuffled with a fixed seed and given opaque ids. The source system, the stratum and
the split are kept in the key file and never sent to the scoring page.

Fixed 30/50 split: the 30 "dev" items may be used to tune the judge rubric; judge-human agreement
is reported on the 50 "test" items only.

A's items are stratified by check outcome, so scores on this set validate the judge; they are not
an estimate of how often A's replies are good. Selection uses predictions and case texts only,
never the golden labels. Refuses to overwrite once human scores exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reply_checks import check_reply  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "golden" / "to_label.jsonl"
HS = ROOT / "data" / "human_scores"
ITEMS = HS / "items.jsonl"
SCORES = HS / "scores_round1.jsonl"
SALT = "ba-human-2026-09-11"
QUOTA = {"A_fail": 15, "A_pass": 30, "B1": 30, "B0": 5}
N_DEV = 30


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _h(*parts) -> str:
    return hashlib.sha256(":".join([SALT, *map(str, parts)]).encode()).hexdigest()


def select(run_dir: Path, out: Path = ITEMS, scores: Path = SCORES) -> list[dict]:
    if scores.exists():
        sys.exit(f"{scores.name} exists: the scoring set is frozen once scoring starts")
    cases = {str(c["case_id"]): c for c in read_jsonl(CASES)}
    preds = {s: {str(r["case_id"]): r for r in read_jsonl(run_dir / f"{s}.jsonl")} for s in ("A", "B1", "B0-auto-all")}
    uniform = sorted((i for i, c in cases.items() if c["stratum"] == "uniform"), key=lambda i: _h(i))
    a_ok = {i: check_reply(preds["A"][i]["reply"], cases[i]["customer_msg"], cases[i]["context"])["passes"]
            for i in uniform if preds["A"][i]["reply"]}      # an empty fail-safe draft has nothing to score
    picked: dict[str, list[str]] = {k: [] for k in QUOTA}
    used: set[str] = set()

    def fill(key: str, pool: list[str], n: int) -> None:
        for i in pool:
            if len(picked[key]) >= n:
                break
            if i not in used:
                picked[key].append(i)
                used.add(i)

    fill("A_fail", [i for i in uniform if a_ok.get(i) is False], QUOTA["A_fail"])
    fill("A_pass", [i for i in uniform if a_ok.get(i) is True], QUOTA["A_pass"] + QUOTA["A_fail"] - len(picked["A_fail"]))
    fill("B1", uniform, QUOTA["B1"])
    fill("B0", uniform, QUOTA["B0"])
    source = {"A_fail": "A", "A_pass": "A", "B1": "B1", "B0": "B0-auto-all"}
    items = []
    for strat, ids in picked.items():
        for i in ids:
            sysname = source[strat]
            c = cases[i]
            items.append({"item_id": _h(i, sysname)[:10], "case_id": i, "source": sysname, "strat": strat,
                          "context": c["context"], "customer_msg": c["customer_msg"],
                          "reply": preds[sysname][i]["reply"]})
    random.Random(SALT).shuffle(items)
    for n, it in enumerate(items):
        it["split"] = "dev" if n < N_DEV else "test"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="select the blind human-scoring set")
    ap.add_argument("--run", type=Path, required=True, help="predictions run with A, B1 and B0-auto-all")
    a = ap.parse_args()
    items = select(a.run)
    by = {}
    for it in items:
        by[it["strat"]] = by.get(it["strat"], 0) + 1
    print(f"{len(items)} items -> {ITEMS.relative_to(ROOT)} | {by} | "
          f"dev {sum(it['split'] == 'dev' for it in items)} / test {sum(it['split'] == 'test' for it in items)}")


if __name__ == "__main__":
    main()

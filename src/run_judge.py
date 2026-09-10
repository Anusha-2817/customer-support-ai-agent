"""Run the LLM judge over replies and write its scores.

    python src/run_judge.py --items data/human_scores/items.jsonl --out artifacts/judge/human80 --local
    python src/run_judge.py --items data/human_scores/items.jsonl --role generator --out artifacts/judge/human80-selfjudge --local
    python src/run_judge.py --run artifacts/predictions/full-v1 --systems A,B1 --stratum uniform --out artifacts/judge/full-v1-uniform --local

Two sources of items:
- --items: the blind human-scoring set, so the judge and the human score exactly the same replies;
- --run: a predictions run; one item per (system, case), with ids like "A:<case_id>". Empty drafts
  (fail-safe escalations) are skipped and counted: there is nothing to grade.
The judge sees the customer's message, earlier turns and the reply only; src/judge.py is blind to
the system and to BA's actual reply. The default role is "judge" (locally llama3.2:3b, a different
model family from the reply writer). --role generator makes qwen2.5:3b, the model that wrote the
agent's replies, act as judge: the self-preference probe. Writes judge.jsonl and run_info.json.
Reads no labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
from judge import Judge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "golden" / "to_label.jsonl"
META = ("source", "case_id", "split", "strat")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def items_from_file(path: Path) -> list[dict]:
    return [{"item_id": r["item_id"], "customer_msg": r["customer_msg"], "context": r["context"], "reply": r["reply"],
             **{k: r[k] for k in META if k in r}} for r in read_jsonl(path)]


def items_from_run(run_dir: Path, systems: list[str], stratum: str | None) -> tuple[list[dict], int]:
    cases = {str(c["case_id"]): c for c in read_jsonl(CASES)}
    items, skipped = [], 0
    for s in systems:
        for r in read_jsonl(run_dir / f"{s}.jsonl"):
            c = cases[str(r["case_id"])]
            if stratum and c["stratum"] != stratum:
                continue
            if not r["reply"]:
                skipped += 1
                continue
            items.append({"item_id": f"{s}:{r['case_id']}", "source": s, "case_id": str(r["case_id"]),
                          "customer_msg": c["customer_msg"], "context": c["context"], "reply": r["reply"]})
    return items, skipped


def run(items: list[dict], out_dir: Path, role: str = "judge", seed: int = 0) -> dict:
    judge = Judge(role=role, seed=seed)
    ok, why = judge.available()
    if not ok:
        raise SystemExit(f"refusing to judge: the {role} model is unavailable ({why})")
    t0 = time.time()
    scores = judge.score_many(items)
    seconds = round(time.time() - t0, 1)
    misses = [s["item_id"] for s in scores if any("CacheMiss" in e for e in s["errors"])]
    if misses:
        raise SystemExit(f"{len(misses)} items missed the offline cache (e.g. {misses[:3]}); nothing written")
    meta = {it["item_id"]: {k: it[k] for k in META if k in it} | {"reply_chars": len(it["reply"])} for it in items}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "judge.jsonl", "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps({**meta[s["item_id"]], **s}, ensure_ascii=False) + "\n")
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "judge": judge.spec, "n_items": len(items),
            "seconds": seconds, "valid": sum(s["valid"] for s in scores), **llm.run_info((role,))}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="run the LLM judge over replies")
    llm.add_mode_args(ap)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--items", type=Path, help="blind human-scoring set (data/human_scores/items.jsonl)")
    src.add_argument("--run", type=Path, help="predictions run directory")
    ap.add_argument("--systems", default="A,B1", help="with --run: systems to judge")
    ap.add_argument("--stratum", choices=["uniform", "targeted"], help="with --run: only this stratum")
    ap.add_argument("--role", default="judge", help="judge (default) or generator (self-preference probe)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    llm.apply_mode_args(a)
    if a.items:
        items, skipped = items_from_file(a.items), 0
    else:
        items, skipped = items_from_run(a.run, [s.strip() for s in a.systems.split(",")], a.stratum)
    items = items[:a.limit] if a.limit else items
    info = run(items, a.out, a.role)
    print(f"{info['n_items']} replies judged by {info['judge']} in {info['seconds']}s -> {a.out} "
          f"| valid {info['valid']}/{info['n_items']}" + (f" | {skipped} empty drafts skipped" if skipped else ""))


if __name__ == "__main__":
    main()

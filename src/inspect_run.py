"""Show a run's cases next to each system's decisions and the reply checks, for human review.

    python src/inspect_run.py artifacts/predictions/<run> [--system A]

Works for golden and dev runs. Reads case texts and predictions only, never labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reply_checks import check_reply, failed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASE_FILES = [ROOT / "data" / "golden" / "to_label.jsonl", ROOT / "data" / "dev" / "dev_cases.jsonl"]


def load_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for path in CASE_FILES:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        cases[str(c["case_id"])] = c
    return cases


def render(run_dir: Path, system: str | None = None) -> str:
    cases = load_cases()
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    out = [f"run {run_dir.name}: {info.get('n_cases')} cases, mode {info.get('mode')}, "
           f"models { {r: m['spec'] for r, m in (info.get('models') or {}).items()} }"]
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.stem in ("judge",) or (system and path.stem != system):
            continue
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        n_fail = 0
        out.append(f"\n=== {path.stem}: {sum(r['needs_human'] for r in recs)}/{len(recs)} escalated, "
                   f"{sum(r.get('valid', True) for r in recs)}/{len(recs)} valid outputs")
        for k, r in enumerate(recs, 1):
            c = cases.get(str(r["case_id"]), {})
            chk = check_reply(r["reply"], c.get("customer_msg", ""), c.get("context"))
            fails = failed(chk)
            n_fail += bool(fails)
            by = ",".join(r.get("escalated_by") or []) or ("guard" if (r.get("guard") or {}).get("escalated") else "")
            decision = f"ESCALATE {r['escalation_reason']} (by {by or 'system'})" if r["needs_human"] else "AUTO"
            signals = " ".join(f"{s}={r[s]}" for s in ("intent_confidence", "tfidf_support", "retrieval_top_score")
                               if r.get(s) is not None)
            out += [f"\n[{k}] {'(thread) ' if c.get('context') else ''}{c.get('customer_msg', '?')[:180]}",
                    f"    intent={r['intent']} | {decision} | {signals}",
                    f"    reply: {r['reply']}",
                    f"    checks: {'pass' if not fails else 'FAIL ' + ', '.join(fails)}"
                    + (f" | errors: {r['errors']}" if r.get("errors") else "")]
        out.append(f"\n{path.stem}: {n_fail}/{len(recs)} drafts fail the reply checks")
    return "\n".join(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="review a predictions run")
    ap.add_argument("run", type=Path)
    ap.add_argument("--system")
    a = ap.parse_args()
    print(render(a.run, a.system))


if __name__ == "__main__":
    main()

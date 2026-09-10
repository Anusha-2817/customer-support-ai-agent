"""Intra-annotator agreement: pick 30 golden cases for a blind re-label, then compare the rounds.

    python src/relabel.py pick     # writes data/golden/relabel_ids.json; reads no labels
    python tools/label_server.py --only-ids data/golden/relabel_ids.json --out data/golden/labels_round2.jsonl --seed 8
    python src/relabel.py agree    # after round 2 -> artifacts/analysis/relabel_agreement.{json,md}

pick: 20 uniform and 10 targeted cases (the strata's shares of 200, rounded), chosen by a salted
hash of case_id, so the choice depends on nothing the labeller did. Cases in the blind
reply-scoring set (data/human_scores/items.jsonl) are left out, so no re-labelled case is one the
labeller has just seen a drafted reply to. Reads no labels, and refuses to run once round-2 labels
exist.

agree: raw agreement and Cohen's kappa for intent, escalation and BA-reply acceptability; reason
agreement on the cases both rounds escalate; the hours between the two labels of each case (the
protocol asks for at least 24). This is the ceiling for any automatic labeller: a system can't be
expected to agree with the labeller more than the labeller agrees with themselves. Refuses to run
without round-2 labels; nothing is substituted for them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden"
CASES = GOLD / "to_label.jsonl"
IDS = GOLD / "relabel_ids.json"
R1 = GOLD / "labels_round1.jsonl"
R2 = GOLD / "labels_round2.jsonl"
ITEMS = ROOT / "data" / "human_scores" / "items.jsonl"
OUT = ROOT / "artifacts" / "analysis"
SALT = "ba-relabel-2026-09-11"
QUOTA = {"uniform": 20, "targeted": 10}
FIELDS = ("intent", "escalate", "ba_reply_ok")
MIN_HOURS = 24


def _h(x: str) -> str:
    return hashlib.sha256(f"{SALT}:{x}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_by_case(path: Path) -> dict[str, dict]:
    return {str(r["case_id"]): r for r in read_jsonl(path)}


def pick(cases_path: Path = CASES, out: Path = IDS, round2: Path = R2, exclude: Path = ITEMS) -> list[str]:
    if round2.exists():
        raise SystemExit(f"{round2} exists: the re-label set is frozen")
    cases = read_jsonl(cases_path)
    scored = {str(r["case_id"]) for r in read_jsonl(exclude)} if exclude.exists() else set()
    ids: list[str] = []
    for stratum, n in QUOTA.items():
        ids += sorted((str(c["case_id"]) for c in cases
                       if c["stratum"] == stratum and str(c["case_id"]) not in scored), key=_h)[:n]
    ids.sort(key=lambda i: _h("order:" + i))
    out.write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")
    return ids


def agree(r1_path: Path = R1, r2_path: Path = R2, ids_path: Path = IDS, out_dir: Path = OUT) -> dict:
    if not r2_path.exists():
        raise SystemExit(f"{r2_path} not found: re-label the 30 cases first (see this file's docstring). "
                         "Nothing is substituted for them.")
    ids = [str(i) for i in json.loads(ids_path.read_text(encoding="utf-8"))]
    r1, r2 = last_by_case(r1_path), last_by_case(r2_path)
    both = [i for i in ids if i in r1 and i in r2]
    res: dict = {"n": len(both), "of": len(ids)}
    for f in FIELDS:
        a, b = [r1[i].get(f) for i in both], [r2[i].get(f) for i in both]
        res[f] = {"agreement": M._r(M.rate(sum(x == y for x, y in zip(a, b)), len(both))), "kappa": M.kappa(a, b)}
    esc = [i for i in both if r1[i]["escalate"] and r2[i]["escalate"]]
    res["reason_when_both_escalate"] = {
        "n": len(esc), "agreement": M._r(M.rate(sum(r1[i].get("reason") == r2[i].get("reason") for i in esc), len(esc)))}
    hours = [(datetime.fromisoformat(r2[i]["ts"]) - datetime.fromisoformat(r1[i]["ts"])).total_seconds() / 3600
             for i in both]
    res["hours_between_rounds"] = {"min": round(min(hours), 1), "median": round(sorted(hours)[len(hours) // 2], 1)} if hours else None
    res["under_min_hours"] = sum(h < MIN_HOURS for h in hours)
    res["changed"] = [{"case_id": i, **{f: [r1[i].get(f), r2[i].get(f)] for f in FIELDS + ("reason",)
                                        if r1[i].get(f) != r2[i].get(f)}}
                      for i in both if any(r1[i].get(f) != r2[i].get(f) for f in FIELDS + ("reason",))]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "relabel_agreement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "relabel_agreement.md").write_text(to_markdown(res), encoding="utf-8")
    return res


def to_markdown(r: dict) -> str:
    L = ["# Intra-annotator agreement (blind re-label)", "", f"Cases re-labelled: {r['n']} of {r['of']}", "",
         "| Field | Agreement | Cohen's kappa |", "|---|---|---|"]
    L += [f"| {f} | {r[f]['agreement']} | {r[f]['kappa']} |" for f in FIELDS]
    rs, hb = r["reason_when_both_escalate"], r["hours_between_rounds"]
    L += ["", f"Escalation reason, when both rounds escalate: agreement {rs['agreement']} (n={rs['n']})",
          f"Hours between rounds: {hb}; cases re-labelled under {MIN_HOURS} h: {r['under_min_hours']}", "",
          f"Changed cases ({len(r['changed'])}):"]
    L += [f"- {c['case_id']}: " + "; ".join(f"{k} {v[0]} -> {v[1]}" for k, v in c.items() if k != "case_id")
          for c in r["changed"]]
    return "\n".join(L) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="blind re-label round: pick the cases, then measure agreement")
    ap.add_argument("step", choices=["pick", "agree"])
    a = ap.parse_args()
    if a.step == "pick":
        ids = pick()
        print(f"{len(ids)} case ids -> {IDS.relative_to(ROOT)}")
    else:
        r = agree()
        print(f"wrote {OUT.relative_to(ROOT) / 'relabel_agreement.md'} | n={r['n']} | "
              + ", ".join(f"{f} kappa {r[f]['kappa']}" for f in FIELDS))


if __name__ == "__main__":
    main()

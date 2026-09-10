"""Judge vs human agreement on the blind scoring set (judge validation, DESIGN.md section 5).

    python src/judge_agreement.py --judge artifacts/judge/human80 --selfjudge artifacts/judge/human80-selfjudge

Refuses to run until the human scores exist (data/human_scores/scores_round1.jsonl): it never
substitutes or simulates human judgments.

Reported on the 50 "test" items; the 30 "dev" items are reported separately because they may be
used to tune the rubric.
- Per dimension: quadratic-weighted kappa and exact agreement between judge and human.
- "Sendable": kappa, accuracy, and the confusion matrix (human rows, judge columns).
- Bias by source system: the judge's mean score minus the human's, for A, B1 and B0 replies.
- Verbosity probe: correlation between reply length and the judge-minus-human gap.
- Self-preference probe (with --selfjudge): the reply-writing model (qwen2.5:3b) acting as judge.
  If it overrates A's replies (its own) relative to the human more than the independent judge
  does, that difference is its self-preference.
- Ceiling: the human's agreement with their own re-scores (scores_round2.jsonl), when present.
Writes agreement.json and agreement.md next to the judge scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from judge import DIMENSIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HS = ROOT / "data" / "human_scores"
DIMS = list(DIMENSIONS)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_by_item(path: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in read_jsonl(path)}


def _mean(xs: list[float]) -> float | None:
    return M._r(float(np.mean(xs))) if xs else None


def gap(judge: dict, human: dict) -> float:
    return float(np.mean([judge[d] - human[d] for d in DIMS]))


def agreement(ids: list[str], human: dict[str, dict], judge: dict[str, dict], items: dict[str, dict]) -> dict:
    ids = [i for i in ids if i in human and i in judge and judge[i].get("valid")]
    out = {"n": len(ids), "dimensions": {}}
    for d in DIMS:
        h, j = [human[i][d] for i in ids], [judge[i][d] for i in ids]
        out["dimensions"][d] = {"kappa_quadratic": M.kappa(h, j, "quadratic") if ids else None,
                                "exact_agreement": M._r(M.rate(sum(a == b for a, b in zip(h, j)), len(ids))),
                                "judge_minus_human": _mean([b - a for a, b in zip(h, j)])}
    hs, js = [human[i]["sendable"] for i in ids], [judge[i]["sendable"] for i in ids]
    out["sendable"] = {"kappa": M.kappa(hs, js) if ids else None,
                       "accuracy": M._r(M.rate(sum(a == b for a, b in zip(hs, js)), len(ids))),
                       "human_rate": _mean([float(x) for x in hs]), "judge_rate": _mean([float(x) for x in js]),
                       "confusion_human_rows_judge_cols": [[sum(1 for a, b in zip(hs, js) if a == r and b == c)
                                                            for c in (True, False)] for r in (True, False)]}
    out["bias_by_source"] = {s: {"n": len(g), "judge_minus_human": _mean([gap(judge[i], human[i]) for i in g])}
                             for s in sorted({items[i]["source"] for i in ids})
                             for g in [[i for i in ids if items[i]["source"] == s]]}
    lengths = [len(items[i]["reply"]) for i in ids]
    gaps = [gap(judge[i], human[i]) for i in ids]
    out["verbosity_corr_length_vs_gap"] = (M._r(float(np.corrcoef(lengths, gaps)[0, 1]))
                                           if len(ids) > 2 and np.std(lengths) > 0 and np.std(gaps) > 0 else None)
    return out


def self_preference(ids: list[str], human: dict, judge: dict, selfj: dict, items: dict) -> dict:
    def bias(scores: dict, source: str) -> float | None:
        g = [gap(scores[i], human[i]) for i in ids
             if i in human and i in scores and scores[i].get("valid") and items[i]["source"] == source]
        return _mean(g)
    out = {k: {"A": bias(s, "A"), "B1": bias(s, "B1")} for k, s in (("independent_judge", judge), ("self_judge", selfj))}
    ind, own = out["independent_judge"], out["self_judge"]
    if None not in (ind["A"], ind["B1"], own["A"], own["B1"]):
        out["self_preference"] = M._r((own["A"] - own["B1"]) - (ind["A"] - ind["B1"]))
        out["reading"] = ("positive = the reply-writing model favours its own replies more than the independent "
                          "judge does, relative to the human")
    return out


def human_ceiling(r1: dict[str, dict], r2: dict[str, dict]) -> dict:
    ids = [i for i in r2 if i in r1]
    return {"n": len(ids), **{d: M.kappa([r1[i][d] for i in ids], [r2[i][d] for i in ids], "quadratic") for d in DIMS},
            "sendable": M.kappa([r1[i]["sendable"] for i in ids], [r2[i]["sendable"] for i in ids])}


def compute(judge_dir: Path, selfjudge_dir: Path | None = None, human_path: Path = HS / "scores_round1.jsonl",
            items_path: Path = HS / "items.jsonl", round2_path: Path = HS / "scores_round2.jsonl") -> dict:
    if not human_path.exists():
        raise SystemExit(f"{human_path} not found: human scores come first "
                         "(python tools/score_server.py). Nothing is substituted for them.")
    items = {r["item_id"]: r for r in read_jsonl(items_path)}
    human = last_by_item(human_path)
    judge = last_by_item(judge_dir / "judge.jsonl")
    selfj = last_by_item(selfjudge_dir / "judge.jsonl") if selfjudge_dir else None
    test = [i for i, r in items.items() if r["split"] == "test"]
    dev = [i for i, r in items.items() if r["split"] == "dev"]
    res = {"human_scored": len(human), "judge": json.loads((judge_dir / "run_info.json").read_text(encoding="utf-8")).get("judge"),
           "test": agreement(test, human, judge, items), "dev": agreement(dev, human, judge, items)}
    if selfj is not None:
        res["self_preference_test"] = self_preference(test, human, judge, selfj, items)
    if round2_path.exists():
        res["human_self_agreement"] = human_ceiling(human, last_by_item(round2_path))
    (judge_dir / "agreement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (judge_dir / "agreement.md").write_text(to_markdown(res), encoding="utf-8")
    return res


def to_markdown(r: dict) -> str:
    L = ["# Judge vs human agreement", "", f"Judge: `{r['judge']}` | human-scored items: {r['human_scored']}", ""]
    for split in ("test", "dev"):
        a = r[split]
        L += [f"## {'Reported: 50 test items' if split == 'test' else 'Dev items (may be used to tune the rubric)'} (n={a['n']})", "",
              "| Dimension | Quadratic kappa | Exact agreement | Judge minus human |", "|---|---|---|---|"]
        L += [f"| {d} | {v['kappa_quadratic']} | {v['exact_agreement']} | {v['judge_minus_human']} |" for d, v in a["dimensions"].items()]
        s = a["sendable"]
        L += ["", f"Sendable: kappa {s['kappa']}, accuracy {s['accuracy']}, human rate {s['human_rate']}, judge rate {s['judge_rate']}; "
              f"confusion (human rows yes/no, judge columns yes/no) {s['confusion_human_rows_judge_cols']}",
              f"Bias by source (judge minus human, mean over dimensions): {a['bias_by_source']}",
              f"Verbosity probe (correlation of reply length with the judge-minus-human gap): {a['verbosity_corr_length_vs_gap']}", ""]
    if "self_preference_test" in r:
        L += ["## Self-preference probe (test items)", "", f"{r['self_preference_test']}", ""]
    if "human_self_agreement" in r:
        L += ["## Ceiling: human self-agreement on re-scored items", "", f"{r['human_self_agreement']}", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="judge vs human agreement")
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--selfjudge", type=Path)
    a = ap.parse_args()
    r = compute(a.judge, a.selfjudge)
    print(f"wrote {a.judge / 'agreement.md'} | test n={r['test']['n']}")


if __name__ == "__main__":
    main()

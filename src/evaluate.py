"""Score a predictions run against golden labels.

    python src/evaluate.py --run artifacts/predictions/<run>
    python src/evaluate.py --run <dir> --labels <file> --smoke      # tests: mock labels / mock models

Subsets, in order of importance:
- headline: the 130 uniform cases. Every headline number comes from here.
- targeted: the 70 enrichment cases, reported separately. Mixing them into headline rates
  without reweighting would distort every rate.
- sensitivity: uniform cases the labeller had NOT seen in the practice round. A check on the
  headline, never a replacement for it.
Only labelled cases are scored; each subset reports how many that was.

Per system and subset: intent accuracy, macro-F1, per-intent P/R/F1 and the confusion matrix;
must-escalate recall, unsafe-auto rate, coverage, escalation precision, reason agreement;
the risk-coverage curve and the best operating point under the 5% unsafe-auto bar; the
deterministic reply checks, both over all replies and over auto-sent replies only (what
would actually reach customers); and the share of valid model outputs. Headline rates carry
95% bootstrap intervals, and the agent is compared with B1 and B0 by paired bootstrap.

Secondary analysis: agreement between the labeller's practice-round escalation decisions
and their final labels, on the practice-exposed cases.

Refuses to report predictions from a mock model unless --smoke is given, and marks the
output as a smoke test when it is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from reply_checks import check_reply  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden"
DEFAULT_LABELS = GOLD / "labels_round1.jsonl"
CASES = GOLD / "to_label.jsonl"
TAXONOMY = GOLD / "taxonomy.json"
PREFILL = GOLD / "prefill_from_practice.jsonl"
MAX_RISK = 0.05
COMPARE = [("A", "B1"), ("A", "B0-auto-all")]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path: Path) -> dict[str, dict]:
    """Last save per case wins (the labelling tool appends edits)."""
    return {str(r["case_id"]): r for r in read_jsonl(path)}


def load_predictions(run_dir: Path) -> dict[str, dict[str, dict]]:
    return {p.stem: {str(r["case_id"]): r for r in read_jsonl(p)}
            for p in sorted(run_dir.glob("*.jsonl")) if p.stem != "judge"}


def score_subset(ids: list[str], preds: dict[str, dict], labels: dict[str, dict], cases: dict[str, dict],
                 intents: list[str], B: int) -> dict:
    ids = [i for i in ids if i in preds and i in labels]
    n = len(ids)
    if not n:
        return {"n": 0}
    P = [preds[i] for i in ids]
    L = [labels[i] for i in ids]
    yt = np.array([l["intent"] for l in L], dtype=object)
    yp = np.array([p["intent"] if p["intent"] in intents else M.INVALID for p in P], dtype=object)
    ge = np.array([bool(l["escalate"]) for l in L])
    pe = np.array([bool(p["needs_human"]) for p in P])
    checks = [check_reply(p["reply"], cases[i]["customer_msg"], cases[i]["context"]) for p, i in zip(P, ids)]
    ok = np.array([c["passes"] for c in checks])
    conf = [p.get("consistency", p.get("intent_confidence")) for p in P]
    auto = ~pe

    def summary(mask: np.ndarray) -> dict:
        sel = [c for c, m in zip(checks, mask) if m]
        k = len(sel)
        return {"n": k, "passes": M._r(M.rate(sum(c["passes"] for c in sel), k)),
                "fabricated": M._r(M.rate(sum(bool(c["fabricated"]) for c in sel), k)),
                "promises": M._r(M.rate(sum(bool(c["promises"]) for c in sel), k)),
                "public_pii_request": M._r(M.rate(sum(c["public_pii_request"] for c in sel), k)),
                "claims_booking_check": M._r(M.rate(sum(c["claims_booking_check"] for c in sel), k)),
                "too_long": M._r(M.rate(sum(c["too_long"] for c in sel), k)),
                "empty": M._r(M.rate(sum(c["empty"] for c in sel), k))}

    curve = M.risk_coverage(ge, pe, conf)
    return {
        "n": n,
        "intent": {**M.intent_metrics(list(yt), list(yp), intents),
                   "accuracy_ci": M.estimate(lambda x: float(np.mean(yt[x] == yp[x])), n, B=B),
                   "macro_f1_ci": M.estimate(lambda x: M.intent_metrics(list(yt[x]), list(yp[x]), intents)["macro_f1"], n, B=B)},
        "escalation": {**M.escalation_metrics(ge, pe),
                       "must_escalate_recall_ci": M.estimate(lambda x: M.rate((ge[x] & pe[x]).sum(), ge[x].sum()), n, B=B),
                       "unsafe_auto_rate_ci": M.estimate(lambda x: M.rate((ge[x] & ~pe[x]).sum(), (~pe[x]).sum()), n, B=B),
                       "coverage_ci": M.estimate(lambda x: float(np.mean(~pe[x])), n, B=B),
                       **M.reason_agreement([l.get("reason") for l in L], [p["escalation_reason"] for p in P], ge, pe)},
        "risk_coverage": {"confidence_source": "consistency" if any("consistency" in p for p in P) else "intent_confidence",
                          **curve, "best_under_bar": M.best_operating_point(curve, MAX_RISK)},
        "reply_checks": {"all_replies": summary(np.ones(n, bool)), "auto_sent_only": summary(auto),
                         "auto_sent_pass_ci": M.estimate(lambda x: M.rate((ok[x] & ~pe[x]).sum(), (~pe[x]).sum()), n, B=B)},
        "valid_outputs": M._r(np.mean([p.get("valid", True) for p in P])),
    }


def compare(ids: list[str], a: dict[str, dict], b: dict[str, dict], labels: dict[str, dict], B: int) -> dict:
    ids = [i for i in ids if i in a and i in b and i in labels]
    n = len(ids)
    yt = np.array([labels[i]["intent"] for i in ids], dtype=object)
    ge = np.array([bool(labels[i]["escalate"]) for i in ids])
    ya, yb = (np.array([s[i]["intent"] for i in ids], dtype=object) for s in (a, b))
    pa, pb = (np.array([bool(s[i]["needs_human"]) for i in ids]) for s in (a, b))
    acc = lambda y: (lambda x: float(np.mean(yt[x] == y[x])))                       # noqa: E731
    unsafe = lambda p: (lambda x: M.rate((ge[x] & ~p[x]).sum(), (~p[x]).sum()))    # noqa: E731
    cover = lambda p: (lambda x: float(np.mean(~p[x])))                             # noqa: E731
    return {"n": n,
            "intent_accuracy": M.paired_difference(acc(ya), acc(yb), n, B=B),
            "unsafe_auto_rate": M.paired_difference(unsafe(pa), unsafe(pb), n, B=B),
            "coverage": M.paired_difference(cover(pa), cover(pb), n, B=B)}


def practice_agreement(labels: dict[str, dict], prefill: list[dict]) -> dict:
    """Secondary: the labeller's practice escalation decision vs their final label, on the
    practice-exposed cases that had a pre-set escalation."""
    pairs = [(bool(p["escalate"]), bool(labels[str(p["case_id"])]["escalate"])) for p in prefill
             if p.get("escalate") is not None and str(p["case_id"]) in labels]
    changed = [r for r in labels.values() if r.get("prefilled_from_practice")]
    return {"n": len(pairs),
            "escalation_agreement": M._r(M.rate(sum(a == b for a, b in pairs), len(pairs))),
            "escalation_kappa": M.kappa([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None,
            "prefilled_labels_saved": len(changed),
            "changed_any_prefilled_field": sum(bool(r.get("changed_from_prefill")) for r in changed)}


def evaluate(run_dir: Path, labels_path: Path = DEFAULT_LABELS, cases_path: Path = CASES,
             smoke: bool = False, B: int = 1000, prefill_path: Path | None = PREFILL) -> dict:
    labels = load_labels(labels_path)
    preds = load_predictions(run_dir)
    if not preds:
        raise SystemExit(f"no predictions in {run_dir}")
    mock = sorted({r["model"] for s in preds.values() for r in s.values() if str(r.get("model", "")).startswith("mock:")})
    if mock and not smoke:
        raise SystemExit(f"predictions come from a mock model {mock}; rerun with --smoke to score them as a test")
    cases = {str(c["case_id"]): c for c in read_jsonl(cases_path)}
    intents = [i["id"] for i in json.loads(TAXONOMY.read_text(encoding="utf-8"))["intents"]]
    uniform = [i for i, c in cases.items() if c["stratum"] == "uniform"]
    targeted = [i for i, c in cases.items() if c["stratum"] == "targeted"]
    unexposed = [i for i in uniform if i in labels and not labels[i].get("prior_exposure_to_ba_reply")]
    subsets = {"headline_uniform": uniform, "targeted": targeted, "sensitivity_uniform_unexposed": unexposed}
    run_info_path = run_dir / "run_info.json"
    results = {
        "smoke_test": smoke, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labels_file": str(labels_path), "labelled": {k: sum(i in labels for i in v) for k, v in subsets.items()},
        "run_info": json.loads(run_info_path.read_text(encoding="utf-8")) if run_info_path.exists() else None,
        "systems": {s: {k: score_subset(v, p, labels, cases, intents, B) for k, v in subsets.items()}
                    for s, p in preds.items()},
        "comparisons_headline": {f"{a} vs {b}": compare(uniform, preds[a], preds[b], labels, B)
                                 for a, b in COMPARE if a in preds and b in preds},
    }
    if prefill_path is not None and prefill_path.exists():
        results["secondary_practice_vs_final"] = practice_agreement(labels, read_jsonl(prefill_path))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "results.md").write_text(to_markdown(results), encoding="utf-8")
    return results


def _fmt(e: dict | None) -> str:
    if not e or e.get("value") is None:
        return "n/a"
    ci = e.get("ci95")
    return f"{e['value']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{e['value']:.2f}"


def to_markdown(r: dict) -> str:
    L = ["# Evaluation results", ""]
    if r["smoke_test"]:
        L += ["> **SMOKE TEST**: mock labels and/or mock model output. These numbers mean nothing.", ""]
    info = r.get("run_info") or {}
    L += [f"Labels: `{r['labels_file']}` - labelled cases per subset: {r['labelled']}",
          f"Run: git `{info.get('git_commit')}`, mode `{info.get('mode')}`, models "
          f"{ {k: v['spec'] for k, v in (info.get('models') or {}).items()} }", ""]
    titles = {"headline_uniform": "Headline: 130 uniform cases",
              "targeted": "Targeted 70 (reported separately, not part of the headline)",
              "sensitivity_uniform_unexposed": "Sensitivity: uniform cases not seen in the practice round"}
    for key, title in titles.items():
        L += [f"## {title}", "", "| System | n | Intent acc | Macro-F1 | Must-escalate recall | Unsafe auto-send | "
              "Coverage | Auto-sent replies passing checks | Valid outputs |", "|---|---|---|---|---|---|---|---|---|"]
        for s, subs in r["systems"].items():
            x = subs[key]
            if not x.get("n"):
                L.append(f"| {s} | 0 | | | | | | | |")
                continue
            L.append(f"| {s} | {x['n']} | {_fmt(x['intent']['accuracy_ci'])} | {_fmt(x['intent']['macro_f1_ci'])} | "
                     f"{_fmt(x['escalation']['must_escalate_recall_ci'])} | {_fmt(x['escalation']['unsafe_auto_rate_ci'])} | "
                     f"{_fmt(x['escalation']['coverage_ci'])} | {_fmt(x['reply_checks']['auto_sent_pass_ci'])} | "
                     f"{x['valid_outputs']:.2f} |")
        L.append("")
    L += ["## Best operating point under the 5% unsafe-auto bar (headline)", "",
          "| System | Confidence source | Threshold | Coverage | Unsafe auto-send |", "|---|---|---|---|---|"]
    for s, subs in r["systems"].items():
        x = subs["headline_uniform"]
        if x.get("n"):
            b = x["risk_coverage"]["best_under_bar"]
            L.append(f"| {s} | {x['risk_coverage']['confidence_source']} | {b['threshold']} | {b['coverage']} | {b['unsafe_auto_rate']} |")
    L += ["", "## Paired comparisons (headline; difference = first minus second)", ""]
    for k, c in r["comparisons_headline"].items():
        L.append(f"- **{k}** (n={c['n']}): " + "; ".join(
            f"{m} {v['difference']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" if v and v.get("ci95") else f"{m} n/a"
            for m, v in c.items() if m != "n"))
    if "secondary_practice_vs_final" in r:
        L += ["", "## Secondary: practice round vs final labels", "", f"{r['secondary_practice_vs_final']}"]
    return "\n".join(L) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="score a predictions run against golden labels")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--smoke", action="store_true", help="allow mock models / mock labels; output is marked as a test")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    r = evaluate(a.run, a.labels, smoke=a.smoke, B=a.bootstrap)
    print(f"wrote {a.run / 'results.md'} | labelled: {r['labelled']}")


if __name__ == "__main__":
    main()

"""Metric tests (src/metrics.py): hand-computed examples, edge cases, and the behaviour of
the bootstrap intervals. Pure computation, no data files. Exit code 1 on any failure.

    python tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import metrics as M  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# ---------------------------------------------------------------- intent
im = M.intent_metrics(["a", "a", "b", "c"], ["a", "b", "b", None], ["a", "b", "c", "d"])
check(im["accuracy"] == 0.5, f"accuracy 2/4 = {im['accuracy']}")
check(im["per_intent"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}, "per-intent P/R/F1 for 'a'")
check(im["per_intent"]["b"]["precision"] == 0.5 and im["per_intent"]["b"]["recall"] == 1.0, "per-intent P/R for 'b'")
check(im["per_intent"]["c"]["f1"] == 0.0 and im["invalid_predictions"] == 1, "an invalid prediction counts as wrong")
check(im["macro_over_intents"] == 3 and im["macro_f1"] == round((2 / 3 + 2 / 3 + 0) / 3, 4),
      f"macro-F1 averages the 3 intents present (unrounded), not the absent 'd': {im['macro_f1']}")
cm = im["confusion"]
check(cm["rows_gold_cols_pred"][-1] == M.INVALID and cm["matrix"][2][-1] == 1, "confusion matrix has an 'invalid' column")

# ------------------------------------------------------------ escalation
em = M.escalation_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 0])
check((em["should_escalate"], em["escalated"], em["auto_sent"], em["missed_escalations"]) == (3, 2, 3, 2), "counts")
check(em["must_escalate_recall"] == 0.3333, f"must-escalate recall 1/3 = {em['must_escalate_recall']}")
check(em["unsafe_auto_rate"] == 0.6667, f"unsafe auto-send 2 of 3 auto-sent = {em['unsafe_auto_rate']}")
check(em["coverage"] == 0.6 and em["escalation_precision"] == 0.5, "coverage 3/5, precision 1/2")
em0 = M.escalation_metrics([0, 0], [1, 1])
check(em0["must_escalate_recall"] is None and em0["unsafe_auto_rate"] is None and em0["coverage"] == 0.0,
      "undefined rates are None, not 0")
ra = M.reason_agreement(["legal_threat", "safety_medical", None], ["legal_threat", "repeated_contact", None],
                        [True, True, False], [True, True, False])
check(ra == {"n_both_escalated": 2, "same_reason": 0.5}, f"reason agreement on jointly escalated cases: {ra}")

# ---------------------------------------------------------- risk-coverage
gold = [True, False, False, True, False, False]
pred = [False, False, False, False, False, True]        # system escalates only the last case
conf = [0.2, 0.9, 0.8, 0.6, None, 0.99]
rc = M.risk_coverage(gold, pred, conf)
covs = [p["coverage"] for p in rc["points"]]
check(covs[0] == round(5 / 6, 4) and covs[-1] == 0.0, f"curve runs from the system's own coverage (5/6) to 0: {covs}")
check(all(a >= b for a, b in zip(covs, covs[1:])), "coverage never rises as the threshold rises")
check(rc["points"][0]["unsafe_auto_rate"] == 0.4, "at the system's own decisions, 2 of 5 auto-sent should have escalated")
best = M.best_operating_point(rc, 0.05)
check(best["unsafe_auto_rate"] == 0.0 and best["coverage"] == round(2 / 6, 4), f"best point under 5% risk: {best}")
check(M.best_operating_point(M.risk_coverage([True], [False], [0.5]), 0.05)["coverage"] == 0.0,
      "if no threshold is safe enough, the operating point is 'escalate everything'")

# ------------------------------------------------------------- bootstrap
rng = np.random.default_rng(1)
small, large = rng.random(100) < 0.3, rng.random(2000) < 0.3
e_small = M.estimate(lambda x: float(np.mean(small[x])), len(small))
e_large = M.estimate(lambda x: float(np.mean(large[x])), len(large))
check(e_small["ci95"][0] <= e_small["value"] <= e_small["ci95"][1], f"interval contains the estimate: {e_small}")
check(e_large["ci95"][1] - e_large["ci95"][0] < e_small["ci95"][1] - e_small["ci95"][0], "more cases, narrower interval")
check(M.estimate(lambda x: float(np.mean(small[x])), len(small), seed=0) == e_small, "fixed seed: reproducible interval")
# Defined on the full sample, but only in resamples that contain both case 0 and case 1
# (about 0.63^2 = 40% of them): too unstable for an interval.
fragile = M.estimate(lambda x: float(np.mean(x < 50)) if ((x == 0).any() and (x == 1).any()) else None, 100)
check(fragile["value"] == 0.5 and fragile["ci95"] is None,
      f"a statistic undefined in most resamples keeps its estimate but gets no interval: {fragile}")
pd_same = M.paired_difference(lambda x: float(np.mean(small[x])), lambda x: float(np.mean(small[x])), 100)
check(pd_same["difference"] == 0.0 and pd_same["ci95"] == [0.0, 0.0], "a system compared with itself differs by exactly 0")

# ------------------------------------------------------------- agreement
check(M.kappa([1, 2, 3], [1, 2, 3]) == 1.0 and M.kappa([2, 2], [2, 2]) == 1.0, "identical scores give kappa 1")
a, b = [1, 2, 3, 3, 2, 1], [1, 2, 2, 3, 3, 1]
check(M.kappa(a, b, "quadratic") > M.kappa(a, b), "quadratic weights credit near-misses on an ordinal scale")

print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

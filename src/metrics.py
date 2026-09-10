"""Evaluation metrics: pure functions over aligned lists. No I/O, no models.

Intent: accuracy, macro-F1, per-intent precision/recall/F1/support, confusion matrix. An
invalid prediction (None, or anything outside the taxonomy) simply counts as wrong. Macro-F1
averages over the intents actually present in the gold labels, and says how many that was.

Escalation (positive class = "should escalate"):
    must_escalate_recall = escalated / should have escalated        (the safety metric)
    unsafe_auto_rate     = should have escalated / auto-sent        (bad posts reaching customers)
    coverage             = auto-sent / all cases                     (work taken off humans)

Risk-coverage: sweep a confidence threshold t. A case is auto-sent only if the system says
"auto" AND its confidence >= t. Each t gives one (coverage, unsafe_auto_rate) point
(Geifman & El-Yaniv, 2017).

Uncertainty: percentile bootstrap over cases (1,000 resamples, fixed seed), and a paired
bootstrap for the difference between two systems scored on the same cases. A statistic that
is undefined in most resamples (e.g. recall with almost no positive cases) gets no interval
rather than a misleading one.

Agreement: Cohen's kappa, quadratic-weighted for ordinal scores (judge vs human).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

INVALID = "__invalid__"
Stat = Callable[[np.ndarray], "float | None"]


def rate(num, den) -> float | None:
    return float(num) / float(den) if den else None


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


# ------------------------------------------------------------------------ intent
def intent_metrics(y_true: Sequence[str], y_pred: Sequence[str | None], labels: list[str]) -> dict:
    yt = list(y_true)
    yp = [p if p in labels else INVALID for p in y_pred]
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    present = [i for i, lab in enumerate(labels) if lab in yt]
    return {
        "n": len(yt),
        "accuracy": _r(rate(sum(t == q for t, q in zip(yt, yp)), len(yt))),
        "macro_f1": _r(np.mean([f[i] for i in present])) if present else None,
        "macro_over_intents": len(present),
        "invalid_predictions": sum(q == INVALID for q in yp),
        "per_intent": {lab: {"precision": _r(p[i]), "recall": _r(r[i]), "f1": _r(f[i]), "support": int(s[i])}
                       for i, lab in enumerate(labels)},
        "confusion": {"rows_gold_cols_pred": labels + [INVALID],
                      "matrix": confusion_matrix(yt, yp, labels=labels + [INVALID]).tolist()},
    }


# -------------------------------------------------------------------- escalation
def escalation_metrics(gold_escalate: Sequence[bool], pred_escalate: Sequence[bool]) -> dict:
    g, p = np.asarray(gold_escalate, bool), np.asarray(pred_escalate, bool)
    auto = ~p
    return {
        "n": int(len(g)),
        "should_escalate": int(g.sum()),
        "escalated": int(p.sum()),
        "auto_sent": int(auto.sum()),
        "missed_escalations": int((g & auto).sum()),
        "must_escalate_recall": _r(rate((g & p).sum(), g.sum())),
        "unsafe_auto_rate": _r(rate((g & auto).sum(), auto.sum())),
        "coverage": _r(rate(auto.sum(), len(g))),
        "escalation_precision": _r(rate((g & p).sum(), p.sum())),
    }


def reason_agreement(gold_reasons: Sequence[str | None], pred_reasons: Sequence[str | None],
                     gold_escalate: Sequence[bool], pred_escalate: Sequence[bool]) -> dict:
    """Among cases both the gold label and the system escalated: how often the reason matches."""
    both = [(g, p) for g, p, ge, pe in zip(gold_reasons, pred_reasons, gold_escalate, pred_escalate) if ge and pe]
    return {"n_both_escalated": len(both), "same_reason": _r(rate(sum(g == p for g, p in both), len(both)))}


# ------------------------------------------------------------------ risk-coverage
def risk_coverage(gold_escalate: Sequence[bool], pred_escalate: Sequence[bool],
                  confidence: Sequence[float | None]) -> dict:
    g = np.asarray(gold_escalate, bool)
    sys_auto = ~np.asarray(pred_escalate, bool)
    c = np.array([-np.inf if x is None else float(x) for x in confidence])
    thresholds = [-np.inf] + sorted(set(c[sys_auto & np.isfinite(c)].tolist())) + [np.inf]
    points = []
    for t in thresholds:
        auto = sys_auto & (c >= t)
        k = int(auto.sum())
        points.append({"threshold": None if t == -np.inf else ("all_escalated" if t == np.inf else round(float(t), 4)),
                       "coverage": _r(rate(k, len(g))), "unsafe_auto_rate": _r(rate((g & auto).sum(), k))})
    pts = sorted(points, key=lambda x: x["coverage"] or 0.0)
    aurc = 0.0
    for a, b in zip(pts, pts[1:]):   # area under the risk-coverage curve (lower is better)
        ra, rb = a["unsafe_auto_rate"] or 0.0, b["unsafe_auto_rate"] or 0.0
        aurc += (b["coverage"] - a["coverage"]) * (ra + rb) / 2
    return {"points": points, "aurc": _r(aurc)}


def best_operating_point(curve: dict, max_risk: float = 0.05) -> dict:
    """Highest coverage whose unsafe-auto rate stays within max_risk (the pre-registered bar)."""
    ok = [p for p in curve["points"] if p["unsafe_auto_rate"] is not None and p["unsafe_auto_rate"] <= max_risk]
    if not ok:
        return {"max_risk": max_risk, "threshold": "all_escalated", "coverage": 0.0, "unsafe_auto_rate": None}
    best = max(ok, key=lambda p: p["coverage"])
    return {"max_risk": max_risk, **best}


# ------------------------------------------------------------------- uncertainty
def bootstrap_ci(stat: Stat, n: int, B: int = 1000, seed: int = 0, alpha: float = 0.05) -> list | None:
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        v = stat(rng.integers(0, n, n))
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    if len(vals) < B / 2:        # undefined in most resamples: an interval would mislead
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [_r(lo), _r(hi)]


def estimate(stat: Stat, n: int, **kw) -> dict:
    """Point estimate on all cases plus its 95% bootstrap interval."""
    v = stat(np.arange(n)) if n else None
    return {"value": _r(v), "ci95": bootstrap_ci(stat, n, **kw) if v is not None else None}


def paired_difference(stat_a: Stat, stat_b: Stat, n: int, B: int = 1000, seed: int = 0) -> dict | None:
    """a - b on the same cases, with a paired bootstrap interval (same resample for both)."""
    a, b = (stat_a(np.arange(n)), stat_b(np.arange(n))) if n else (None, None)
    if a is None or b is None:
        return None
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        x, y = stat_a(idx), stat_b(idx)
        if x is not None and y is not None:
            diffs.append(x - y)
    if len(diffs) < B / 2:
        return {"difference": _r(a - b), "ci95": None}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": _r(a - b), "ci95": [_r(lo), _r(hi)],
            "share_of_resamples_below_zero": _r(np.mean(np.asarray(diffs) < 0))}


# --------------------------------------------------------------------- agreement
def kappa(a: Sequence, b: Sequence, weights: str | None = None) -> float | None:
    if len(a) == 0:
        return None
    if len(set(a) | set(b)) == 1:    # perfect agreement on a single value: sklearn returns nan
        return 1.0
    return _r(cohen_kappa_score(list(a), list(b), weights=weights))

"""Golden-set sampling. Every candidate comes from the eval universe (after the
temporal cut), never from the retrieval pool.

    python src/sample_golden.py uniform     # 130 cases, unbiased
    python src/sample_golden.py targeted    # 70 cases, needs artifacts/silver/eval_universe.jsonl
    python src/sample_golden.py build       # merge both into data/golden/to_label.jsonl

Two strata, recorded in the output but hidden from the labeller:
- uniform  (130): unbiased estimates for the headline numbers.
- targeted  (70): enrichment, so rare intents and likely escalations have enough cases
  to measure at all. Reported separately; mixing it into headline rates without
  reweighting would distort them.

Targeted allocation, in order:
1. rare intents: top each intent up to 10 cases across both strata (at most 40 picks),
   found by silver label (nearest centroid + curated mapping);
2. 10 long threads (at least two earlier turns);
3. likely escalations for the rest of the 70, found by keyword triggers. These matter
   most: must-escalate recall needs enough positive cases for a usable confidence
   interval. A first version sent unused rare-intent slots to random top-ups (26 of 70);
   now they go here.
Neither heuristic is ground truth, and the triggers are deliberately NOT the agent's
guard rules: if the guard's own rules chose the test cases, it would look better than
it is.

Selection is by a salted hash of case_id, not by row position. Row-position sampling
(df.sample) reshuffles the entire sample whenever an upstream fix adds or drops a
single row - fixing the sign-off cleaner moved one row and changed all 130 picks.
With hashing, a case is picked or not on its own merits, so the golden set survives
later data fixes. Once real labels exist, every step refuses to run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "processed" / "eval_universe.jsonl"
SILVER = ROOT / "artifacts" / "silver" / "eval_universe.jsonl"
GOLD = ROOT / "data" / "golden"
TAXONOMY = GOLD / "taxonomy.json"
UNIFORM = GOLD / "sample_uniform.jsonl"
TARGETED = GOLD / "sample_targeted.jsonl"
TO_LABEL = GOLD / "to_label.jsonl"
LABELS = GOLD / "labels_round1.jsonl"
SALT = "ba-golden-2026-09-10"

N_UNIFORM, N_TARGETED = 130, 70
RARE_BUDGET, THREAD_BUDGET = 40, 10   # likely escalations get the rest
RARE_TARGET = 10   # aim for at least this many cases per intent across both strata

# Sampling heuristics only: they decide which cases get a closer look, never a label.
TRIGGERS = {
    "legal":  r"\b(?:lawyers?|solicitors?|sue|suing|legal action|small claims|court|caa|ombudsman|regulator)\b",
    "money":  r"\b(?:compensation|eu ?261|refunds?|reimburse\w*|expenses|claims?)\b",
    "safety": r"\b(?:ill|sick|medical|injur\w*|hospital|wheelchair|pregnan\w*|disab\w*|unsafe|emergency|died|death|funeral)\b",
    "repeat": r"\b(?:still (?:no|waiting|not)|third time|no (?:reply|response)|(?:\d+|two|three|several) (?:weeks|days|calls|times))\b",
}


def frozen_guard() -> None:
    if LABELS.exists():
        sys.exit(f"{LABELS.name} exists: the golden set is frozen once labelling starts")


def load_eval() -> pd.DataFrame:
    # convert_dates=False: pandas would otherwise parse "created_at" and write it
    # back as epoch milliseconds.
    return pd.read_json(EVAL, lines=True, convert_dates=False)


def stable_pick(df: pd.DataFrame, n: int, salt: str = SALT) -> pd.DataFrame:
    """The n cases with the smallest salted hash of their case_id."""
    key = df["case_id"].astype(str).map(
        lambda c: hashlib.sha256(f"{salt}:{c}".encode()).hexdigest())
    return df.assign(_h=key).sort_values("_h").head(n).drop(columns="_h")


def write(df: pd.DataFrame, path: Path) -> None:
    GOLD.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def uniform(n: int = N_UNIFORM) -> pd.DataFrame:
    frozen_guard()
    s = stable_pick(load_eval(), n).assign(stratum="uniform", target_reason=None)
    write(s, UNIFORM)
    return s


def targeted() -> pd.DataFrame:
    frozen_guard()
    if not SILVER.exists():
        sys.exit("artifacts/silver/eval_universe.jsonl not found: run src/silver.py first")
    intents = [i["id"] for i in json.loads(TAXONOMY.read_text(encoding="utf-8"))["intents"]]
    ev = load_eval().merge(pd.read_json(SILVER, lines=True)[["case_id", "silver_intent"]], on="case_id")
    taken = set(pd.read_json(UNIFORM, lines=True)["case_id"])
    counts = ev[ev.case_id.isin(taken)].silver_intent.value_counts()
    picks: list[pd.DataFrame] = []

    def take(pool: pd.DataFrame, n: int, reason: str) -> int:
        pool = pool[~pool.case_id.isin(taken)]
        got = stable_pick(pool, n, salt=f"{SALT}:{reason}")
        taken.update(got.case_id)
        picks.append(got.assign(target_reason=reason))
        return len(got)

    # 1. rare intents (by silver label): top each up towards RARE_TARGET cases
    need = {i: max(0, RARE_TARGET - int(counts.get(i, 0))) for i in intents}
    total = sum(need.values())
    if total > RARE_BUDGET:
        need = {i: max(1, round(v * RARE_BUDGET / total)) if v else 0 for i, v in need.items()}
    for i, n in need.items():
        if n:
            take(ev[ev.silver_intent == i], n, f"rare_intent:{i}")

    # 2. long threads: at least two earlier turns to read
    take(ev[ev.context.map(len) >= 2], THREAD_BUDGET, "long_thread")

    # 3. likely escalations get the rest. Trigger types share it; a type that runs out
    #    (legal threats are rare) passes its share on to the others.
    text = ev.customer_msg.str.lower()
    pools = {name: ev[text.str.contains(pat, regex=True)] for name, pat in TRIGGERS.items()}
    left = N_TARGETED - sum(len(p) for p in picks)
    while left > 0:
        open_types = [n for n, p in pools.items() if (~p.case_id.isin(taken)).any()]
        if not open_types:
            break
        share = max(1, left // len(open_types))
        for n in open_types:
            left -= take(pools[n], min(share, left), f"trigger:{n}")
            if left <= 0:
                break

    if left > 0:                 # every pool ran dry: top up so the stratum size is fixed
        take(ev, left, "top_up")
    t = pd.concat(picks).head(N_TARGETED).drop(columns="silver_intent").assign(stratum="targeted")
    write(t, TARGETED)
    return t


def build() -> pd.DataFrame:
    frozen_guard()
    u = pd.read_json(UNIFORM, lines=True, convert_dates=False)
    t = pd.read_json(TARGETED, lines=True, convert_dates=False)
    both = pd.concat([u, t], ignore_index=True)
    assert both.case_id.is_unique, "a case appears in both strata"
    assert set(both.case_id) <= set(load_eval().case_id), "a golden case is not from the eval universe"
    write(both, TO_LABEL)
    return both


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "uniform"
    if step == "uniform":
        s = uniform()
        print(f"uniform: {len(s)} cases -> {UNIFORM.relative_to(ROOT)} | with prior context: "
              f"{(s.context.map(len) > 0).mean():.0%} | days covered: {s.created_at.str[:10].nunique()}")
    elif step == "targeted":
        t = targeted()
        print(f"targeted: {len(t)} cases -> {TARGETED.relative_to(ROOT)}")
        print(t.target_reason.value_counts().to_string())
    elif step == "build":
        b = build()
        print(f"to_label: {len(b)} cases -> {TO_LABEL.relative_to(ROOT)} | "
              + " | ".join(f"{k} {v}" for k, v in b.stratum.value_counts().items()))
    else:
        sys.exit(f"unknown step {step!r} (uniform | targeted | build)")

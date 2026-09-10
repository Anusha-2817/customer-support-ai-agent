"""A 30-case development set drawn from the retrieval pool, for prompt and rule tuning.

    python src/dev_set.py                     # writes data/dev/dev_cases.jsonl
    python src/run_systems.py --dev --systems A --local --run-name dev-try1

Golden cases are for evaluation only: anything tuned while looking at them would be fitted to
its own test. So prompt and rule changes are tried on these pool cases instead. Three cases
per silver intent (stable hash pick), so every intent is represented. When a system runs on
the dev set, retrieval uses the pool minus these 30 cases, so a dev case can't retrieve its
own BA reply.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "processed" / "pool.jsonl"
SILVER_POOL = ROOT / "artifacts" / "silver" / "pool.jsonl"
OUT = ROOT / "data" / "dev" / "dev_cases.jsonl"
PER_INTENT = 3
SALT = "ba-dev-2026-09-10"


def _hash(case_id) -> str:
    return hashlib.sha256(f"{SALT}:{case_id}".encode()).hexdigest()


def build(out: Path | None = None) -> pd.DataFrame:
    out = out or OUT
    pool = pd.read_json(POOL, lines=True, convert_dates=False)
    silver = pd.read_json(SILVER_POOL, lines=True)[["case_id", "silver_intent"]]
    d = pool.merge(silver, on="case_id", validate="one_to_one")
    d = d.assign(_h=d.case_id.map(_hash)).sort_values("_h")
    dev = (d.groupby("silver_intent", group_keys=False).head(PER_INTENT)
           .drop(columns="_h").assign(stratum="dev").sort_values("case_id"))
    out.parent.mkdir(parents=True, exist_ok=True)
    dev.to_json(out, orient="records", lines=True, force_ascii=False)
    return dev


def dev_ids(path: Path | None = None) -> set[int]:
    return set(pd.read_json(path or OUT, lines=True, convert_dates=False).case_id)


def retriever_without_dev(ids: set[int] | None = None):
    """A retriever over the pool minus the dev cases."""
    from retrieval import Retriever
    ids = ids if ids is not None else dev_ids()
    pool = pd.read_json(POOL, lines=True, convert_dates=False)
    return Retriever(pool[~pool.case_id.isin(ids)])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = build()
    print(f"dev set: {len(df)} pool cases -> {OUT.relative_to(ROOT)} | "
          + " | ".join(f"{k} {v}" for k, v in df.silver_intent.value_counts().sort_index().items()))

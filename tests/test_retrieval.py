"""Retrieval tests (src/retrieval.py).

    python tests/test_retrieval.py

Runs in --offline mode: the pool and eval embeddings were cached by the silver-label step,
so no model is loaded and nothing is computed or downloaded. Writes nothing.
Exit code 1 on any failure.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402
from retrieval import Retriever  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


llm.set_mode("offline")
pool = pd.read_json(ROOT / "data/processed/pool.jsonl", lines=True, convert_dates=False)
ev = pd.read_json(ROOT / "data/processed/eval_universe.jsonl", lines=True, convert_dates=False)
sil_pool = dict(pd.read_json(ROOT / "artifacts/silver/pool.jsonl", lines=True)[["case_id", "silver_intent"]].values)
sil_eval = dict(pd.read_json(ROOT / "artifacts/silver/eval_universe.jsonl", lines=True)[["case_id", "silver_intent"]].values)

r = Retriever(pool)
q = ev.sample(300, random_state=0)
res = r.search(q.customer_msg.tolist(), k=5)
check(all(len(x) == 5 for x in res), "5 neighbours per query")
check(all(all(a["score"] >= b["score"] for a, b in zip(x, x[1:])) for x in res), "neighbours sorted by similarity")
got = {n["case_id"] for x in res for n in x}
check(got <= set(pool.case_id) and not got & set(ev.case_id), "neighbours come only from the pool, never the eval universe")
shares = pd.Series(list(sil_pool.values())).value_counts(normalize=True)
chance = float((shares ** 2).sum())
agree = sum(sil_pool[n["case_id"]] == sil_eval[cid] for cid, x in zip(q.case_id, res) for n in x) / (5 * len(q))
check(agree > 2 * chance, f"neighbours share the query's silver intent {agree:.0%} of the time (chance {chance:.0%})")
check(all({"case_id", "customer_msg", "brand_reply", "score"} <= set(n) for x in res for n in x),
      "each neighbour carries the customer message, BA's reply and the score")
check(r.search(["anything"], k=0) == [[]], "k=0 retrieves nothing (the no-retrieval ablation)")

print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

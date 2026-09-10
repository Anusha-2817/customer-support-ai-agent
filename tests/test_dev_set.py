"""Dev-set tests (src/dev_set.py).

    python tests/test_dev_set.py

30 pool cases, 3 per silver intent, disjoint from the golden cases, stable across rebuilds,
and a dev retriever that can't return a dev case's own BA reply. Builds into a temporary
directory; embeddings replay offline from the cache. Exit code 1 on any failure.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dev_set as D  # noqa: E402
import llm  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


tmp = Path(tempfile.mkdtemp(prefix="devset_"))
try:
    df = D.build(tmp / "dev.jsonl")
    again = D.build(tmp / "dev2.jsonl")
    counts = df.silver_intent.value_counts()
    check(len(df) == 30 and set(counts) == {3} and len(counts) == 10, f"30 cases, 3 per silver intent: {dict(counts)}")
    pool_ids = set(pd.read_json(ROOT / "data/processed/pool.jsonl", lines=True, convert_dates=False).case_id)
    golden_ids = {json.loads(l)["case_id"] for l in open(ROOT / "data/golden/to_label.jsonl", encoding="utf-8")}
    check(set(df.case_id) <= pool_ids, "every dev case comes from the retrieval pool")
    check(not set(df.case_id) & golden_ids, "no dev case is a golden case")
    check(list(df.case_id) == list(again.case_id), "rebuilding gives the same 30 cases")
    check((df.stratum == "dev").all() and {"customer_msg", "context", "brand_reply"} <= set(df.columns), "dev cases carry what the runner needs")

    llm.set_mode("offline")
    r = D.retriever_without_dev(set(df.case_id))
    hits = {n["case_id"] for x in r.search(df.customer_msg.tolist(), k=5) for n in x}
    check(len(r.pool) == len(pool_ids) - 30 and not hits & set(df.case_id),
          "the dev retriever excludes the 30 dev cases, so none retrieves its own BA reply")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

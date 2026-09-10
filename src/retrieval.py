"""Retrieval of similar past BA cases (design C3).

    from retrieval import Retriever
    r = Retriever()                       # embeds the pool once; cached on disk after that
    r.search(["my bag never arrived"], k=5)

The index holds every retrieval-pool case (before the temporal cut), matched on the customer
message and returning the (customer message, BA reply) pair. The eval universe is never
indexed, so a golden case can't retrieve its own answer. Brute-force cosine in numpy:
17,809 x 384 is one matrix product per query batch, so no vector database is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "processed" / "pool.jsonl"


class Retriever:
    def __init__(self, pool: pd.DataFrame | None = None):
        if pool is None:
            pool = pd.read_json(POOL, lines=True, convert_dates=False)
        self.pool = pool.reset_index(drop=True)
        self.spec = llm.spec_for("embedding")
        self.M = llm.embed(self.pool.customer_msg.tolist())   # unit-length rows

    def search(self, queries: list[str], k: int = 5, batch: int = 256) -> list[list[dict]]:
        """Top-k pool cases per query, most similar first."""
        if k <= 0:
            return [[] for _ in queries]
        k = min(k, len(self.pool))
        out: list[list[dict]] = []
        for start in range(0, len(queries), batch):
            S = llm.embed(queries[start:start + batch]) @ self.M.T
            top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
            for row, idx in zip(S, top):
                idx = idx[np.argsort(-row[idx])]
                out.append([{"case_id": int(self.pool.case_id[i]),
                             "customer_msg": self.pool.customer_msg[i],
                             "brand_reply": self.pool.brand_reply[i],
                             "score": round(float(row[i]), 4)} for i in idx])
        return out

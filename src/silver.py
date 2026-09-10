"""Silver intent labels without an LLM: assign each message to its nearest taxonomy
centroid, then apply the human-curated cluster -> intent mapping.

    python src/silver.py [--local | --offline]

Writes artifacts/silver/pool.jsonl and artifacts/silver/eval_universe.jsonl with
case_id, cluster, silver_intent and centroid_sim (cosine similarity to that centroid).

Silver labels are never ground truth. They are used for two things only:
- training the TF-IDF baseline (B1) on the retrieval pool;
- enriching the golden set with rare intents (the targeted stratum).
Known flaws, reported rather than hidden: every member of a mixed cluster gets that
cluster's single intent, and an intent without a cluster of its own gets no silver
labels at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAX = ROOT / "artifacts" / "taxonomy"
OUT = ROOT / "artifacts" / "silver"
PROCESSED = ROOT / "data" / "processed"


def load_mapping() -> tuple[np.ndarray, dict[int, str]]:
    m = json.loads((TAX / "cluster_to_intent.json").read_text(encoding="utf-8"))
    C = np.load(TAX / "centroids.npy")
    # Centroids only mean something in the embedding space they were computed in.
    if m["embedding"] != llm.spec_for("embedding"):
        sys.exit(f"centroids were built with {m['embedding']}, but the active embedding model is "
                 f"{llm.spec_for('embedding')}; re-run taxonomy propose + finalize first")
    if C.shape[0] != m["k"]:
        sys.exit("centroids.npy does not match cluster_to_intent.json")
    return C, {int(k): v for k, v in m["mapping"].items()}


def assign(texts: list[str], C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centroid by cosine similarity (rows and centroids are unit length)."""
    S = llm.embed(texts) @ C.T
    return S.argmax(axis=1), S.max(axis=1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="silver intent labels from the curated cluster mapping")
    llm.add_mode_args(ap)
    llm.apply_mode_args(ap.parse_args())
    C, mapping = load_mapping()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("pool", "eval_universe"):
        df = pd.read_json(PROCESSED / f"{name}.jsonl", lines=True, convert_dates=False)
        print(f"{name}: embedding {len(df):,} messages with {llm.spec_for('embedding')} ...", flush=True)
        cl, sim = assign(df.customer_msg.tolist(), C)
        out = pd.DataFrame({"case_id": df.case_id, "cluster": cl,
                            "silver_intent": [mapping[int(c)] for c in cl], "centroid_sim": sim.round(4)})
        out.to_json(OUT / f"{name}.jsonl", orient="records", lines=True)
        share = out.silver_intent.value_counts(normalize=True)
        print("  " + " | ".join(f"{k} {v:.1%}" for k, v in share.items()), flush=True)
    (OUT / "run_info.json").write_text(json.dumps(llm.run_info(("embedding",)), indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Smoke test for the taxonomy pipeline, fully offline.

    python tests/test_taxonomy.py

Runs `propose` with a mock embedder (manual naming, then mock-LLM naming) and
`finalize` on missing, invalid, valid and frozen curation files. Everything goes to a
temporary directory; the real artifacts/, data/golden/ and cache/ are never touched.
The mock vectors are random, so the clusters are meaningless: this checks mechanics,
not quality. Exit code 1 on any failure.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402
import providers as P  # noqa: E402
import taxonomy as tx  # noqa: E402

REAL = [ROOT / "artifacts" / "taxonomy", ROOT / "data" / "golden" / "taxonomy.json"]
real_before = {str(p): p.exists() for p in REAL}
tmp = Path(tempfile.mkdtemp(prefix="taxonomy_smoke_"))
tx.OUT = tmp / "taxonomy"
tx.GOLD_TAXONOMY = tmp / "golden" / "taxonomy.json"
tx.GOLD_LABELS = tmp / "golden" / "labels_round1.jsonl"
tx.N_SAMPLE = 1000
K = 12
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def exits(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


def propose(naming: str) -> None:
    tx.propose(argparse.Namespace(mode="local", profile=None, naming=naming, k=K))


def write_curation(intents: list[dict], mapping: dict[int, str]) -> None:
    cur = json.loads((tx.OUT / "curation_template.json").read_text(encoding="utf-8"))
    cur["intents"] = intents
    for c, v in mapping.items():
        cur["clusters"][str(c)]["intent"] = v
    (tx.OUT / "curation.json").write_text(json.dumps(cur), encoding="utf-8")


def fin(force: bool = False) -> None:
    tx.finalize(argparse.Namespace(force=force))


llm.use_provider("embedding", P.MockEmbedder(dims=32))
try:
    # -- propose, manual naming ------------------------------------------------------
    propose("manual")
    files = ["sweep.csv", "clusters.json", "centroids.npy", "sample_assignments.jsonl",
             "proposal.md", "curation_template.json"]
    check(all((tx.OUT / f).exists() for f in files), "propose writes all six outputs")
    meta = json.loads((tx.OUT / "clusters.json").read_text(encoding="utf-8"))
    check(meta["naming"] == "manual" and meta["k"] == K, "manual naming recorded, k honoured")
    check(meta["run_info"]["models"]["embedding"]["spec"] == "mock:hash-embed", "embedding model recorded")
    md = (tx.OUT / "proposal.md").read_text(encoding="utf-8")
    check("No model named anything" in md and md.count("name: ______") == K and "suggested name" not in md,
          "manual report leaves every name blank and claims no model naming")
    tpl = json.loads((tx.OUT / "curation_template.json").read_text(encoding="utf-8"))
    check(len(tpl["clusters"]) == K and all(v["intent"] == "" for v in tpl["clusters"].values())
          and tpl["_suggestions_by"] is None, "template has no pre-filled intents")

    # -- finalize ----------------------------------------------------------------------
    check(exits(fin), "finalize refuses without curation.json")
    write_curation([{"id": "Flight Disruption", "name": "x", "definition": "y"}], {3: "nonexistent"})
    check(exits(fin) and not tx.GOLD_TAXONOMY.exists(), "invalid curation is rejected and nothing is written")
    ids = [f"intent_{i}" for i in range(8)] + ["other_non_actionable"]
    good = [{"id": i, "name": i.replace("_", " "), "definition": f"The customer wants {i}."} for i in ids]
    write_curation(good, {c: ids[c % len(ids)] for c in range(K)})
    fin()
    tax = json.loads(tx.GOLD_TAXONOMY.read_text(encoding="utf-8"))
    c2i = json.loads((tx.OUT / "cluster_to_intent.json").read_text(encoding="utf-8"))
    check(len(tax["intents"]) == 9 and len(c2i["mapping"]) == K, "valid curation writes taxonomy and cluster mapping")
    tx.GOLD_LABELS.write_text("{}\n", encoding="utf-8")
    check(exits(fin), "finalize refuses once labelling has started")
    fin(force=True)
    check(tx.GOLD_TAXONOMY.exists(), "--force overrides the freeze")

    # -- propose, LLM naming via a mock namer ----------------------------------------
    def namer(messages):
        text = messages[-1]["content"]
        if text.startswith("Rules for the taxonomy"):
            cs = sorted({int(m) for m in re.findall(r"Cluster (\d+) \(", text)})
            return json.dumps({"intents": [{"id": f"s{j}", "name": f"Suggested {j}", "definition": "d",
                                            "clusters": cs[j::9], "boundary_notes": "b"} for j in range(9)]})
        return json.dumps({"name": "mock name", "definition": "The customer wants x.", "coherent": True,
                           "mixed_with": []})

    llm.use_provider("namer", P.MockProvider(namer))
    propose("llm")
    md = (tx.OUT / "proposal.md").read_text(encoding="utf-8")
    tpl = json.loads((tx.OUT / "curation_template.json").read_text(encoding="utf-8"))
    check("suggestions by `mock:v1`" in md and "Suggested intents" in md and md.count("suggested name: mock name") == K,
          "LLM naming is labelled as suggestions and attributed to the model")
    check(tpl["_suggestions_by"] == "mock:v1" and all(v["intent"] for v in tpl["clusters"].values()),
          "template carries attributed suggestions")
    check((tx.OUT / "curation.json").exists(), "an existing curation.json is left untouched")

    # -- --naming llm with no model available ------------------------------------------
    llm._overrides.pop("namer")
    ok, why = llm.llm_available("namer")
    if not ok:
        check(exits(lambda: propose("llm")), f"--naming llm fails loudly without a model ({why})")
finally:
    llm.clear_overrides()
    shutil.rmtree(tmp, ignore_errors=True)

check({str(p): p.exists() for p in REAL} == real_before, "real artifacts/ and data/golden/ untouched")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

"""Run systems over the golden cases and write their predictions.

    python src/run_systems.py --systems B0-escalate-all,B0-auto-all,B1 --offline
    python src/run_systems.py --systems A --limit 5 --local        # first real-model check

Cases come from data/golden/to_label.jsonl: the cases only, never the labels. Predictions go
to artifacts/predictions/<run>/<system>.jsonl, with run_info.json recording the mode, every
model spec and the git commit, so local, mock and paid runs can't be confused.

Systems: B0-escalate-all, B0-auto-all, B1, A, A-no-retrieval, A-no-guard.

Safeguards:
- agent systems refuse to start if the generator model is unavailable, instead of writing
  a file of fail-safe escalations that would look like results;
- a run where any case hit an offline cache miss is aborted and nothing is written;
- --consistency N adds self-consistency confidence: N extra samples at temperature 0.7, and
  the share that agree with the main answer on intent and escalation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
from agent import Agent  # noqa: E402
from baselines import B0, B1  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "golden" / "to_label.jsonl"
PRED_DIR = ROOT / "artifacts" / "predictions"
ALL_SYSTEMS = ["B0-escalate-all", "B0-auto-all", "B1", "A", "A-no-retrieval", "A-no-guard"]


def load_cases(path: Path = CASES, limit: int | None = None, ids: set[str] | None = None) -> list[dict]:
    """Golden cases in file order (uniform stratum first), without BA's reply."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    cases = [{"case_id": str(r["case_id"]), "customer_msg": r["customer_msg"], "context": r["context"],
              "stratum": r["stratum"]} for r in rows]
    if ids:
        cases = [c for c in cases if c["case_id"] in ids]
    return cases[:limit] if limit else cases


def build(name: str, retriever=None):
    if name == "B0-escalate-all":
        return B0(escalate_all=True)
    if name == "B0-auto-all":
        return B0(escalate_all=False)
    if name == "B1":
        return B1(retriever=retriever)
    if name == "A":
        return Agent(retriever=retriever)
    if name == "A-no-retrieval":
        return Agent(k=0)
    if name == "A-no-guard":
        return Agent(use_guard=False, retriever=retriever)
    raise ValueError(f"unknown system {name!r}; choose from {ALL_SYSTEMS}")


def add_consistency(agent: Agent, cases: list[dict], records: list[dict], n: int) -> None:
    """Self-consistency (Wang et al., 2022): share of n sampled answers that agree with the main
    answer on both intent and the model's escalation decision."""
    samples = []
    for i in range(1, n + 1):
        sampler = Agent(k=agent.k, use_guard=agent.use_guard, retriever=agent._retriever,
                        temperature=0.7, seed=i, name=agent.name)
        samples.append(sampler.run_many(cases))
    for j, rec in enumerate(records):
        agree = sum(s[j]["intent"] == rec["intent"] and s[j]["llm_needs_human"] == rec["llm_needs_human"]
                    for s in samples)
        rec["consistency"] = round(agree / n, 4)
        rec["consistency_samples"] = n


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(systems: list[str], cases: list[dict], out_dir: Path, consistency: int = 0, retriever=None) -> dict:
    agents = [s for s in systems if s.startswith("A")]
    if agents:
        ok, why = llm.llm_available("generator")
        if not ok:
            raise SystemExit(f"refusing to run {agents}: the generator model is unavailable ({why})")
    needs_retrieval = any(s in ("B1", "A", "A-no-guard") for s in systems)
    if needs_retrieval and retriever is None:
        from retrieval import Retriever
        retriever = Retriever()
    outputs: dict[str, list[dict]] = {}
    timings: dict[str, float] = {}
    for name in systems:
        system = build(name, retriever)
        t0 = time.time()
        records = system.run_many(cases)
        if consistency and isinstance(system, Agent):
            add_consistency(system, cases, records, consistency)
        timings[name] = round(time.time() - t0, 1)
        misses = [r["case_id"] for r in records if any("CacheMiss" in e for e in r.get("errors", []))]
        if misses:
            raise SystemExit(f"{name}: {len(misses)} cases missed the offline cache (e.g. {misses[:3]}); nothing written")
        stratum = {c["case_id"]: c["stratum"] for c in cases}
        for r in records:
            r["case_id"] = str(r["case_id"])
            r["stratum"] = stratum[r["case_id"]]
        outputs[name] = records
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in outputs.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(), "n_cases": len(cases),
            "systems": systems, "seconds_per_system": timings, "consistency_samples": consistency,
            **llm.run_info(("embedding", "generator"))}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="run systems over the golden cases")
    llm.add_mode_args(ap)
    ap.add_argument("--systems", default="B0-escalate-all,B0-auto-all,B1,A",
                    help=f"comma-separated, from {', '.join(ALL_SYSTEMS)}")
    ap.add_argument("--limit", type=int, help="only the first N cases (uniform stratum first)")
    ap.add_argument("--ids", type=Path, help="JSON list of case_ids to run")
    ap.add_argument("--run-name", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--consistency", type=int, default=0, help="extra samples for self-consistency (agent only)")
    a = ap.parse_args()
    llm.apply_mode_args(a)
    systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    ids = {str(x) for x in json.loads(a.ids.read_text(encoding="utf-8"))} if a.ids else None
    cases = load_cases(limit=a.limit, ids=ids)
    out = PRED_DIR / a.run_name
    info = run(systems, cases, out, a.consistency)
    print(f"{len(cases)} cases x {len(systems)} systems -> {out}")
    for s in systems:
        print(f"  {s:16s} {info['seconds_per_system'][s]:>7.1f}s")
    print("models:", {r: m["spec"] for r, m in info["models"].items()})


if __name__ == "__main__":
    main()

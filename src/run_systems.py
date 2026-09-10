"""Run systems over the golden cases (or the dev set) and write their predictions.

    python src/run_systems.py --systems B0-escalate-all,B0-auto-all,B1 --offline
    python src/run_systems.py --systems A --limit 5 --local          # first real-model check
    python src/run_systems.py --dev --systems A,A+gate --local       # prompt/rule tuning on pool cases

Golden cases come from data/golden/to_label.jsonl: the cases only, never the labels. With --dev
the 30 pool cases in data/dev/dev_cases.jsonl are used instead, with a retriever that excludes
them. Predictions go to artifacts/predictions/<run>/<system>.jsonl, with run_info.json recording
the mode, every model spec and the git commit, so local, mock and paid runs can't be confused.

Systems: B0-escalate-all, B0-auto-all, B1, A, A+gate, A-no-retrieval, A-no-guard.
A and A+gate send identical prompts, so after A has run, A+gate is served entirely from the cache.

Risk signals added to every agent prediction, for the risk-coverage curve (the model's own
confidence came back as 1.0 on every smoke-test case, so it can't rank anything):
- tfidf_support: the probability the TF-IDF intent model gives the agent's chosen intent;
- retrieval_top_score: similarity of the closest past case (recorded by the agent).

Safeguards:
- agent systems refuse to start if the generator model is unavailable, instead of writing
  a file of fail-safe escalations that would look like results;
- a run where any case hit an offline cache miss is aborted and nothing is written;
- --consistency N adds self-consistency confidence (N extra samples at temperature 0.7).
  Off by default: on a laptop CPU it multiplies the run time by N + 1.
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
DEV_CASES = ROOT / "data" / "dev" / "dev_cases.jsonl"
PRED_DIR = ROOT / "artifacts" / "predictions"
ALL_SYSTEMS = ["B0-escalate-all", "B0-auto-all", "B1", "A", "A+gate", "A-no-retrieval", "A-no-guard"]
USES_RETRIEVER = {"B1", "A", "A+gate", "A-no-guard"}


def load_cases(path: Path = CASES, limit: int | None = None, ids: set[str] | None = None) -> list[dict]:
    """Cases in file order (golden: uniform stratum first), without BA's reply."""
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
    if name == "A+gate":
        return Agent(reply_gate=True, retriever=retriever)
    if name == "A-no-retrieval":
        return Agent(k=0)
    if name == "A-no-guard":
        return Agent(use_guard=False, retriever=retriever)
    raise ValueError(f"unknown system {name!r}; choose from {ALL_SYSTEMS}")


def add_risk_signals(records: list[dict], cases: list[dict], clf) -> None:
    """tfidf_support: TF-IDF's probability for the agent's intent (0 if the intent is invalid)."""
    P = clf.pipe.predict_proba([c["customer_msg"] for c in cases])
    classes = list(clf.pipe.classes_)
    for rec, probs in zip(records, P):
        rec["tfidf_support"] = round(float(probs[classes.index(rec["intent"])]), 4) if rec["intent"] in classes else 0.0
        rec["tfidf_agrees"] = rec["intent"] == classes[int(probs.argmax())]


def add_consistency(agent: Agent, cases: list[dict], records: list[dict], n: int) -> None:
    """Self-consistency (Wang et al., 2022): share of n sampled answers that agree with the main
    answer on both intent and the model's escalation decision."""
    samples = []
    for i in range(1, n + 1):
        sampler = Agent(k=agent.k, use_guard=agent.use_guard, reply_gate=agent.reply_gate,
                        retriever=agent._retriever, temperature=0.7, seed=i, name=agent.name)
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
    if USES_RETRIEVER & set(systems) and retriever is None:
        from retrieval import Retriever
        retriever = Retriever()
    clf = None
    if agents:
        import tfidf_intent
        clf = tfidf_intent.train_on_silver()
    outputs: dict[str, list[dict]] = {}
    timings: dict[str, float] = {}
    for name in systems:
        system = build(name, retriever)
        t0 = time.time()
        records = system.run_many(cases)
        if isinstance(system, Agent):
            add_risk_signals(records, cases, clf)
            if consistency:
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
            "case_set": sorted({c["stratum"] for c in cases}), "systems": systems,
            "seconds_per_system": timings, "consistency_samples": consistency,
            **llm.run_info(("embedding", "generator"))}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="run systems over the golden cases or the dev set")
    llm.add_mode_args(ap)
    ap.add_argument("--systems", default="B0-escalate-all,B0-auto-all,B1,A",
                    help=f"comma-separated, from {', '.join(ALL_SYSTEMS)}")
    ap.add_argument("--dev", action="store_true", help="use the 30-case pool dev set instead of the golden cases")
    ap.add_argument("--limit", type=int, help="only the first N cases (golden: uniform stratum first)")
    ap.add_argument("--ids", type=Path, help="JSON list of case_ids to run")
    ap.add_argument("--run-name", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--consistency", type=int, default=0, help="extra samples for self-consistency (agent only)")
    a = ap.parse_args()
    llm.apply_mode_args(a)
    systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    ids = {str(x) for x in json.loads(a.ids.read_text(encoding="utf-8"))} if a.ids else None
    retriever = None
    if a.dev:
        import dev_set
        if not DEV_CASES.exists():
            sys.exit("data/dev/dev_cases.jsonl not found: run python src/dev_set.py first")
        retriever = dev_set.retriever_without_dev() if USES_RETRIEVER & set(systems) else None
    cases = load_cases(DEV_CASES if a.dev else CASES, limit=a.limit, ids=ids)
    out = PRED_DIR / a.run_name
    info = run(systems, cases, out, a.consistency, retriever)
    print(f"{len(cases)} {'dev' if a.dev else 'golden'} cases x {len(systems)} systems -> {out}")
    for s in systems:
        print(f"  {s:16s} {info['seconds_per_system'][s]:>7.1f}s")
    print("models:", {r: m["spec"] for r, m in info["models"].items()})


if __name__ == "__main__":
    main()

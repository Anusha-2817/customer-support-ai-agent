"""End-to-end harness test: run every system over all 200 golden cases with a mock model,
score them against SYNTHETIC labels, and check the numbers behave.

    python tests/test_harness_e2e.py

The real labels file (data/golden/labels_round1.jsonl) is fenced off: while this test runs,
any attempt to open it, for reading or writing, is recorded and raises. The test passes only
if there were zero attempts. (Its timestamp is deliberately not checked: the labelling tool
may be appending to it at the same time.) Synthetic labels are random (seeded), not derived
from any human label. Everything is written to a temporary directory. Embeddings replay
offline from the cache. Exit code 1 on any failure.
"""
import builtins
import io
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_LABELS = (ROOT / "data" / "golden" / "labels_round1.jsonl").resolve()
FENCE_HITS: list[str] = []

_open, _io_open = builtins.open, io.open


def fenced_open(file, *args, **kwargs):
    try:
        hit = Path(file).resolve() == REAL_LABELS
    except TypeError:
        hit = False
    if hit:
        FENCE_HITS.append(str(file))
        raise AssertionError("the harness tried to open the real golden labels file")
    return _open(file, *args, **kwargs)


builtins.open = io.open = fenced_open

sys.path.insert(0, str(ROOT / "src"))
import guard  # noqa: E402
import llm  # noqa: E402
import providers as P  # noqa: E402
import run_systems as RS  # noqa: E402
from evaluate import evaluate  # noqa: E402
from inspect_run import render  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def mock_agent(messages):
    msg = messages[-1]["content"].split("Customer message:\n")[1].split("\n")[0].lower()
    intent = "baggage" if "bag" in msg else "flight_disruption" if ("delay" in msg or "cancel" in msg) else "service_complaint"
    # every seventh message gets a draft with an invented fee, so the reply gate has work to do
    reply = ("Your extra bag costs £15." if len(msg) % 7 == 0 else
             "We're sorry to hear this. Please DM us and we'll look into it for you.")
    return json.dumps({"intent": intent, "intent_confidence": 1.0, "needs_human": "refund" in msg,
                       "escalation_reason": "compensation_dispute" if "refund" in msg else None, "reply": reply})


tmp = Path(tempfile.mkdtemp(prefix="harness_e2e_"))
pred_dir_before = sorted((ROOT / "artifacts" / "predictions").glob("*")) if (ROOT / "artifacts" / "predictions").exists() else []
try:
    llm.set_mode("offline")
    llm.use_provider("generator", P.MockProvider(mock_agent))
    cases = RS.load_cases()
    check(len(cases) == 200 and sum(c["stratum"] == "uniform" for c in cases) == 130, "200 golden cases, 130 uniform")
    check(all("brand_reply" not in c for c in cases), "the runner never loads BA's actual replies")
    run_dir = tmp / "run"
    info = RS.run(RS.ALL_SYSTEMS, cases, run_dir, consistency=2)
    check(all((run_dir / f"{s}.jsonl").exists() for s in RS.ALL_SYSTEMS) and (run_dir / "run_info.json").exists(),
          f"one predictions file per system ({len(RS.ALL_SYSTEMS)}) plus run_info.json")
    check(info["models"]["generator"]["spec"] == "mock:v1" and info["mode"] == "offline", "run_info records mode and models")
    a_recs = [json.loads(l) for l in open(run_dir / "A.jsonl", encoding="utf-8")]
    g_recs = [json.loads(l) for l in open(run_dir / "A+gate.jsonl", encoding="utf-8")]
    check(all(0 <= r["tfidf_support"] <= 1 and isinstance(r["retrieval_top_score"], float) for r in a_recs),
          "cheap risk signals recorded: TF-IDF support and top retrieval similarity")
    check(all(0 <= r["consistency"] <= 1 for r in a_recs), "self-consistency still available when asked for")
    check(all(g["reply"] == a["reply"] and (g["needs_human"] or not a["needs_human"]) for a, g in zip(a_recs, g_recs))
          and sum(g["reply_gate"]["escalated"] for g in g_recs) > 0,
          f"A+gate: same drafts as A, only adds escalations ({sum(g['reply_gate']['escalated'] for g in g_recs)} gated)")

    # synthetic labels: random, seeded, loosely tied to the guard so escalation metrics have signal
    rng = random.Random(0)
    intents = [i["id"] for i in json.loads((ROOT / "data/golden/taxonomy.json").read_text(encoding="utf-8"))["intents"]]
    uniform_ids = [c["case_id"] for c in cases if c["stratum"] == "uniform"]
    exposed = set(rng.sample(uniform_ids, 57))
    labels_path = tmp / "synthetic_labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for c in cases:
            hits = guard.check(c["customer_msg"], c["context"])["hits"]
            esc = bool(hits) or rng.random() < 0.1
            f.write(json.dumps({"case_id": c["case_id"], "intent": rng.choice(intents), "escalate": esc,
                                "reason": (hits[0]["reason"] if hits else "unclear_request") if esc else None,
                                "ba_reply_ok": "yes", "prior_exposure_to_ba_reply": c["case_id"] in exposed,
                                "prefilled_from_practice": c["case_id"] in exposed,
                                "changed_from_prefill": ["escalate"] if c["case_id"] in exposed and rng.random() < .2 else []}) + "\n")
    prefill_path = tmp / "synthetic_prefill.jsonl"
    with open(prefill_path, "w", encoding="utf-8") as f:
        for cid in sorted(exposed):
            f.write(json.dumps({"case_id": cid, "escalate": rng.random() < 0.3}) + "\n")

    try:
        evaluate(run_dir, labels_path, B=100, prefill_path=prefill_path)
        check(False, "mock-model predictions are refused without --smoke")
    except SystemExit as e:
        check("mock" in str(e), "mock-model predictions are refused without --smoke")

    res = evaluate(run_dir, labels_path, smoke=True, B=200, prefill_path=prefill_path)
    check(res["labelled"] == {"headline_uniform": 130, "targeted": 70, "sensitivity_uniform_unexposed": 73},
          f"headline 130 uniform, targeted 70 separate, sensitivity drops the 57 exposed: {res['labelled']}")
    sysres = res["systems"]
    check(set(sysres) == set(RS.ALL_SYSTEMS), "every system scored")
    hl = {s: sysres[s]["headline_uniform"] for s in sysres}
    lab = {r["case_id"]: r for r in (json.loads(l) for l in open(labels_path, encoding="utf-8"))}
    base = sum(lab[i]["escalate"] for i in uniform_ids) / 130
    e_auto, e_all = hl["B0-auto-all"]["escalation"], hl["B0-escalate-all"]["escalation"]
    check(e_auto["coverage"] == 1.0 and e_auto["must_escalate_recall"] == 0.0 and e_auto["unsafe_auto_rate"] == round(base, 4),
          f"B0-auto-all: full coverage, zero recall, unsafe rate = base rate {base:.3f}")
    check(e_all["coverage"] == 0.0 and e_all["must_escalate_recall"] == 1.0 and e_all["unsafe_auto_rate"] is None,
          "B0-escalate-all: zero coverage, full recall, nothing auto-sent")
    check(hl["A-no-guard"]["escalation"]["coverage"] >= hl["A"]["escalation"]["coverage"] >= hl["A+gate"]["escalation"]["coverage"],
          "each safety layer only removes coverage: A-no-guard >= A >= A+gate")
    # Synthetic escalations = guard hits plus 10% random ones the guard can't see, so B1
    # (guard-only) should catch exactly the guard-hit share, with no false alarms.
    esc_u = [c for c in cases if c["stratum"] == "uniform" and lab[c["case_id"]]["escalate"]]
    fired = sum(bool(guard.check(c["customer_msg"], c["context"])["hits"]) for c in esc_u)
    b1 = hl["B1"]["escalation"]
    check(b1["must_escalate_recall"] == round(fired / len(esc_u), 4) and b1["escalation_precision"] == 1.0,
          f"B1 recall = the guard's share of escalations ({fired}/{len(esc_u)}), precision 1.0 by construction")
    cis = [(v["value"], v["ci95"]) for s in sysres.values() for sub in s.values() if sub.get("n")
           for grp in ("intent", "escalation") for k, v in sub[grp].items() if k.endswith("_ci") and v["ci95"]]
    check(cis and all(lo <= val <= hi for val, (lo, hi) in cis), f"all {len(cis)} intervals contain their estimate")
    rc_a = hl["A"]["risk_coverage"]
    check({"tfidf_support", "retrieval_top_score", "intent_confidence", "consistency"} <= set(rc_a),
          f"a risk-coverage curve per available signal: {sorted(rc_a)}")
    check(all(c["best_under_bar"]["threshold"] == "all_escalated" or c["best_under_bar"]["unsafe_auto_rate"] <= 0.05
              for s in hl.values() for c in s["risk_coverage"].values()), "every operating point respects the 5% bar")
    check(hl["A+gate"]["reply_checks"]["auto_sent_only"]["passes"] == 1.0 and hl["A"]["reply_checks"]["auto_sent_only"]["passes"] < 1.0,
          "A auto-sends drafts that fail the checks; A+gate's auto-sent drafts pass them (by construction)")
    rc = {s: hl[s]["reply_checks"]["all_replies"] for s in hl}
    check(rc["B1"]["fabricated"] > 0, f"verbatim B1 replies carry invented specifics ({rc['B1']['fabricated']:.0%})")
    check(set(res["comparisons_headline"]) == {"A vs B1", "A vs B0-auto-all", "A+gate vs A"}, "paired comparisons on the headline set")
    check(res["secondary_practice_vs_final"]["n"] == 57, "practice-vs-final agreement reported as a secondary analysis")
    md = (run_dir / "results.md").read_text(encoding="utf-8")
    check(md.startswith("# Evaluation results") and "SMOKE TEST" in md and "Headline: 130 uniform" in md
          and "by construction" in md, "results.md: smoke banner, headline first, and the gate's by-construction caveat")
    view = render(run_dir, "A+gate")
    check("=== A+gate" in view and "drafts fail the reply checks" in view, "inspect_run renders a run for review")

    llm.use_provider("generator", P.MockProvider(lambda m: (_ for _ in ()).throw(llm.CacheMiss("not cached"))))
    try:
        RS.run(["A"], cases[:3], tmp / "miss")
        check(False, "an offline cache miss aborts the run")
    except SystemExit as e:
        check("cache" in str(e) and not (tmp / "miss").exists(), "an offline cache miss aborts the run and writes nothing")
    llm.clear_overrides()
    llm.set_mode("offline")
    ok, why = llm.llm_available("generator")
    if not ok:
        try:
            RS.run(["A"], cases[:3], tmp / "nomodel")
            check(False, "the agent refuses to run without its model")
        except SystemExit as e:
            check("refusing" in str(e) and not (tmp / "nomodel").exists(),
                  f"the agent refuses to run without its model ({why[:60]})")
finally:
    llm.clear_overrides()
    builtins.open, io.open = _open, _io_open
    shutil.rmtree(tmp, ignore_errors=True)

check(not FENCE_HITS, f"the harness made zero attempts to open the real labels file ({len(FENCE_HITS)} attempts)")
now = sorted((ROOT / "artifacts" / "predictions").glob("*")) if (ROOT / "artifacts" / "predictions").exists() else []
check(now == pred_dir_before, "nothing written to artifacts/predictions")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

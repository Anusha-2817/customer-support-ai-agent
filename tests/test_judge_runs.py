"""Tests for the judge runner and the judge-vs-human agreement script, with a mock judge and
SYNTHETIC human scores in a temporary directory.

    python tests/test_judge_runs.py

The real human scores (data/human_scores/scores_*.jsonl) and the golden labels are fenced off:
any attempt to open them fails the test. Nothing here stands in for real human judgments; the
synthetic scores only exercise the code. Exit code 1 on any failure.
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
FENCED = {(ROOT / "data" / "golden" / "labels_round1.jsonl").resolve(),
          (ROOT / "data" / "human_scores" / "scores_round1.jsonl").resolve(),
          (ROOT / "data" / "human_scores" / "scores_round2.jsonl").resolve()}
FENCE_HITS: list[str] = []
_open = builtins.open


def fenced(file, *args, **kwargs):
    try:
        hit = Path(file).resolve() in FENCED
    except TypeError:
        hit = False
    if hit:
        FENCE_HITS.append(str(file))
        raise AssertionError("tried to open a fenced file")
    return _open(file, *args, **kwargs)


builtins.open = io.open = fenced
sys.path.insert(0, str(ROOT / "src"))
import judge_agreement as JA  # noqa: E402
import llm  # noqa: E402
import providers as P  # noqa: E402
import run_judge as RJ  # noqa: E402
from judge import DIMENSIONS  # noqa: E402

failures: list[str] = []
CHAT = ROOT / "cache" / "chat.jsonl"
chat_before = CHAT.stat().st_size if CHAT.exists() else None


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def mock_judge(bump: int = 0):
    """Scores depend on the reply length (so they vary); bump shifts every score (bias)."""
    seen = []

    def responder(messages):
        seen.append(messages[-1]["content"])
        rep = messages[-1]["content"].split("Draft reply to grade:\n")[1].split("\n")[0]
        s = min(3, max(1, 1 + len(rep) % 3 + bump))
        return json.dumps({"rationale": "x", **{d: s for d in DIMENSIONS}, "sendable": s == 3})
    return P.MockProvider(responder), seen


tmp = Path(tempfile.mkdtemp(prefix="judge_runs_"))
try:
    llm.set_mode("offline")
    prov, seen = mock_judge()
    llm.use_provider("judge", prov)
    items = RJ.items_from_file(ROOT / "data" / "human_scores" / "items.jsonl")
    info = RJ.run(items, tmp / "human80")
    js = [json.loads(l) for l in open(tmp / "human80" / "judge.jsonl", encoding="utf-8")]
    check(info["n_items"] == 80 and len(js) == 80 and all(j["valid"] for j in js), "the judge scores all 80 blind items")
    check(all({"source", "split", "reply_chars", "judge"} <= set(j) for j in js), "scores carry the item metadata for analysis")
    check(not any(src in p for p in seen for src in ("B0-auto-all", "\"B1\"", "source")), "the judge's prompts name no system")
    run_items, skipped = RJ.items_from_run(ROOT / "artifacts/predictions/full-v1", ["A", "B1"], "uniform")
    check(len(run_items) + skipped == 260 and all(i["item_id"].split(":")[0] in ("A", "B1") for i in run_items),
          f"predictions mode: 130 uniform cases x 2 systems ({skipped} empty drafts skipped)")

    prov2, _ = mock_judge(bump=1)
    llm.use_provider("generator", prov2)
    RJ.run(items, tmp / "selfjudge", role="generator")

    try:
        JA.compute(tmp / "human80", human_path=tmp / "missing.jsonl")
        check(False, "agreement refuses to run without human scores")
    except SystemExit as e:
        check("human scores come first" in str(e), "agreement refuses to run without human scores (nothing substituted)")

    # SYNTHETIC human scores for code testing only: the judge's own scores, perturbed.
    rng = random.Random(0)
    human = tmp / "human_synthetic.jsonl"
    with open(human, "w", encoding="utf-8") as f:
        for j in js:
            h = {d: min(3, max(1, j[d] + (rng.choice([-1, 1]) if rng.random() < 0.25 else 0))) for d in DIMENSIONS}
            f.write(json.dumps({"item_id": j["item_id"], **h, "sendable": j["sendable"] if rng.random() > .1 else not j["sendable"]}) + "\n")
    res = JA.compute(tmp / "human80", tmp / "selfjudge", human_path=human,
                     items_path=ROOT / "data/human_scores/items.jsonl", round2_path=tmp / "none.jsonl")
    check(res["test"]["n"] == 50 and res["dev"]["n"] == 30, "agreement is reported on the 50 test items, dev separately")
    kq = [v["kappa_quadratic"] for v in res["test"]["dimensions"].values()]
    check(all(k is not None and 0 < k <= 1 for k in kq), f"per-dimension quadratic kappa computed: {kq}")
    check(res["test"]["sendable"]["kappa"] is not None and sum(map(sum, res["test"]["sendable"]["confusion_human_rows_judge_cols"])) == 50,
          "sendable kappa and a 2x2 confusion over the 50 test items")
    check(set(res["test"]["bias_by_source"]) <= {"A", "B1", "B0-auto-all"}, "bias reported per source system")
    sp = res["self_preference_test"]
    check(sp["self_judge"]["A"] is not None and "self_preference" in sp, f"self-preference probe computed: {sp.get('self_preference')}")
    same = JA.human_ceiling({j["item_id"]: j for j in js}, {j["item_id"]: j for j in js[:25]})
    check(same["n"] == 25 and all(same[d] == 1.0 for d in DIMENSIONS), "identical re-scores give a ceiling of kappa 1")
    check((tmp / "human80" / "agreement.md").exists(), "agreement.md written next to the judge scores")
finally:
    llm.clear_overrides()
    builtins.open = io.open = _open
    shutil.rmtree(tmp, ignore_errors=True)

check(not FENCE_HITS, f"never opened the golden labels or the real human scores ({len(FENCE_HITS)} attempts)")
check((CHAT.stat().st_size if CHAT.exists() else None) == chat_before, "mock judge outputs never reached the cache")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

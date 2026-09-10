"""Judge interface tests (src/judge.py) with a mock judge model.

    python tests/test_judge.py

Checks score validation, that results map back to the right item despite the shuffled order,
that the prompt is blind (no system name, no BA reply), that a position-biased judge is caught
by the two-order pairwise check, and how an unavailable local judge is reported. No real
model runs; nothing is written. Exit code 1 on any failure.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402
import providers as P  # noqa: E402
from judge import DIMENSIONS, Judge  # noqa: E402

failures: list[str] = []
CHAT_CACHE = ROOT / "cache" / "chat.jsonl"
chat_before = CHAT_CACHE.read_bytes() if CHAT_CACHE.exists() else None


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def reply_in(prompt: str) -> str:
    return prompt.split("Draft reply to grade:\n")[1].split("\n")[0]


def deterministic_scores(messages):
    """Scores are a function of the reply's length, so we can verify each result's mapping."""
    rep = reply_in(messages[-1]["content"])
    s = 1 + len(rep) % 3
    return json.dumps({"rationale": "ok", **{d: s for d in DIMENSIONS}, "sendable": s == 3})


seen: list[str] = []
llm.set_mode("offline")
llm.use_provider("judge", P.MockProvider(lambda m: (seen.append(m[-1]["content"]), deterministic_scores(m))[1]))
items = [{"item_id": f"i{k}", "customer_msg": f"customer message {k}", "context": [], "reply": "r" * (10 + k),
          "system": "B1", "brand_reply": "SECRET BA REPLY"} for k in range(8)]
res = Judge(seed=0).score_many(items)
check([r["item_id"] for r in res] == [it["item_id"] for it in items], "results come back in the caller's order")
check(all(r["relevance"] == 1 + len(it["reply"]) % 3 for r, it in zip(res, items)),
      "each score belongs to its own item despite the shuffled sending order")
check(all(r["valid"] and r["judge"] == "mock:v1" for r in res), "valid scores, judge model recorded")
check(not any("B1" in p or "SECRET BA REPLY" in p for p in seen), "blind: no system name and no BA reply in any prompt")
check(all(f"customer message {k}" in "".join(seen) for k in range(8)), "the customer's message is in the prompt")

for payload, expect, label in [
    ({"rationale": "x", **{d: 5 for d in DIMENSIONS}, "sendable": "maybe"}, ["relevance", "sendable"], "out-of-range scores"),
    ({"rationale": "x", "relevance": 3}, ["groundedness", "sendable"], "missing fields"),
]:
    llm.use_provider("judge", P.MockProvider(lambda m, p=payload: json.dumps(p)))
    r = Judge().score_many(items[:1])[0]
    check(not r["valid"] and all(any(e.startswith(x) for e in r["errors"]) for x in expect), f"{label} are rejected: {r['errors']}")
llm.use_provider("judge", P.MockProvider(lambda m: json.dumps({"rationale": "x", **{d: "3" for d in DIMENSIONS}, "sendable": "yes"})))
r = Judge().score_many(items[:1])[0]
check(r["valid"] and r["relevance"] == 3 and r["sendable"] is True, "'3' and 'yes' are coerced")


def pair_parts(prompt: str) -> tuple[str, str]:
    a = re.search(r"Reply A:\n(.*?)\n\nReply B:", prompt, re.S).group(1)
    b = re.search(r"Reply B:\n(.*?)\n\nReturn", prompt, re.S).group(1)
    return a, b


item = {"customer_msg": "my bag is lost", "context": []}
llm.use_provider("judge", P.MockProvider(lambda m: json.dumps({"rationale": "x", "better": "A"})))
pw = Judge().pairwise(item, "Please DM us.", "Sorry.")
check(pw["winner"] == "inconsistent" and not pw["consistent"], f"a judge that always picks 'A' is caught: {pw['verdicts']}")
llm.use_provider("judge", P.MockProvider(
    lambda m: json.dumps({"rationale": "x", "better": "A" if "DM" in pair_parts(m[-1]["content"])[0] else "B"})))
pw = Judge().pairwise(item, "Please DM us.", "Sorry.")
check(pw["winner"] == "a" and pw["consistent"], "a judge with a real preference wins in both orders")

llm.clear_overrides()
llm.set_mode("local")
ok, why = Judge().available()
check(isinstance(ok, bool) and (ok or "llama3.2:3b" in why or "Ollama" in why),
      f"local judge status is reported: {'available' if ok else why}")
llm.set_mode("offline")
check("openai" not in sys.modules, "the openai package was never imported")
check((CHAT_CACHE.read_bytes() if CHAT_CACHE.exists() else None) == chat_before, "mock outputs never reached the cache")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

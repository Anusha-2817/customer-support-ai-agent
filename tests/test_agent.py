"""Agent and baseline tests, end to end through the provider interface, with no real LLM.

    python tests/test_agent.py

The generator role is a MockProvider, so every step runs for real (guard, retrieval, prompt,
structured output, validation) except the model itself. Embeddings are replayed in --offline
mode from the cache (the golden cases were embedded by the silver-label step). One test uses
the real local profile to show a missing Ollama fails safe. Golden data is read-only; nothing
is written, no paid call is possible. Exit code 1 on any failure.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402
import providers as P  # noqa: E402
from agent import Agent  # noqa: E402
from baselines import B0, B1, FIXED_REPLY  # noqa: E402
from retrieval import Retriever  # noqa: E402

failures: list[str] = []
FIELDS = ("intent", "intent_confidence", "needs_human", "escalation_reason", "reply")
CHAT_CACHE = ROOT / "cache" / "chat.jsonl"
chat_before = CHAT_CACHE.read_bytes() if CHAT_CACHE.exists() else None


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def mock(reply_json=None, log=None):
    """A generator that answers with fixed JSON (or text) and records every prompt it sees."""
    def responder(messages):
        if log is not None:
            log.append(messages)
        if reply_json is not None:
            return reply_json if isinstance(reply_json, str) else json.dumps(reply_json)
        msg = messages[-1]["content"].split("Customer message:\n")[1].split("\n")[0].lower()
        intent = "baggage" if "bag" in msg else "flight_disruption" if ("delay" in msg or "cancel" in msg) else "other_non_actionable"
        return json.dumps({"intent": intent, "intent_confidence": 0.7, "needs_human": False,
                           "escalation_reason": None, "reply": "Sorry to hear this. Please DM us and we'll help."})
    return P.MockProvider(responder)


golden = [json.loads(l) for l in open(ROOT / "data/golden/to_label.jsonl", encoding="utf-8")]
cases = [{"case_id": g["case_id"], "customer_msg": g["customer_msg"], "context": g["context"]} for g in golden[:20]]
llm.set_mode("offline")
retriever = Retriever()

# ------------------------------------------------------------------ agent, happy path
print("-- agent")
log: list = []
llm.use_provider("generator", mock(log=log))
out = Agent(retriever=retriever).run_many(cases)
check(len(out) == 20 and all(all(f in o for f in FIELDS) for o in out), "20 golden cases -> 20 results with the five output fields")
check(all(o["valid"] and o["model"] == "mock:v1" for o in out), "valid structured output, model recorded")
check(all(len(o["retrieved"]) == 5 for o in out) and all(m[-1]["content"].count("\n   BA: ") == 5 for m in log),
      "5 retrieved BA replies reach each prompt")
leaks = [g["case_id"] for g, m in zip(golden[:20], log)
         if len(g["brand_reply"]) > 40 and g["brand_reply"][:40] in m[-1]["content"]]
check(not leaks, "BA's actual reply to the case never appears in the agent's prompt")
check(all(o["needs_human"] == (o["llm_needs_human"] or bool(o["guard"]["hits"])) for o in out),
      "final escalation = model decision OR guard (monotone)")

# ------------------------------------------------------------------ guard and ablations
threat = {"case_id": "t-legal", "customer_msg": "Lost my bag again, I'll be speaking to my solicitor", "context": []}
o = Agent(k=0).run(threat)
check(o["needs_human"] and o["escalation_reason"] == "legal_threat" and o["guard"]["escalated"] and not o["llm_needs_human"],
      "guard escalates a legal threat the model auto-handled")
o = Agent(k=0, use_guard=False).run(threat)
check(not o["needs_human"] and o["system"] == "A-no-retrieval-no-guard", "without the guard the model's decision stands (ablation)")
log.clear()
Agent(k=0).run({"case_id": "t", "customer_msg": "hello", "context": []})
check("Past British Airways replies" not in log[-1][-1]["content"], "k=0 sends no exemplars (no-retrieval ablation)")

# ------------------------------------------------------------------ bad model output
for label, payload, expect in [
    ("not JSON at all", "Sure! Happy to help.", "no JSON"),
    ("an intent outside the taxonomy", {"intent": "banana", "intent_confidence": 0.9, "needs_human": False,
                                        "escalation_reason": None, "reply": "Hi"}, "invalid intent"),
    ("a reason while saying no human is needed", {"intent": "baggage", "intent_confidence": 0.9, "needs_human": False,
                                                  "escalation_reason": "legal_threat", "reply": "Hi"}, "escalated"),
]:
    llm.use_provider("generator", mock(payload))
    o = Agent(k=0).run({"case_id": "t", "customer_msg": "hello there", "context": []})
    check(o["needs_human"] and not o["valid"] and any(expect in e for e in o["errors"]),
          f"{label}: escalated and recorded as invalid ({o['errors'][0][:60]})")
llm.use_provider("generator", mock('{"intent": "baggage", "intent_confidence": "high", "needs_human": "false", '
                                   '"escalation_reason": "null", "reply": "Sorry, DM us"}'))
o = Agent(k=0).run({"case_id": "t", "customer_msg": "my bag", "context": []})
check(o["intent"] == "baggage" and o["needs_human"] is False and o["intent_confidence"] is None and not o["valid"],
      "loosely typed fields are coerced where safe, and the bad one is recorded")
llm.use_provider("generator", mock({"intent": "baggage", "intent_confidence": 0.55, "needs_human": False,
                                    "escalation_reason": None, "reply": "Sorry, please DM us"}))
o = Agent(k=0, min_confidence=0.8).run({"case_id": "t", "customer_msg": "my bag", "context": []})
check(o["needs_human"] and o["escalation_reason"] == "unclear_request" and o["low_confidence"],
      "below the confidence threshold the case goes to a human as an unclear request")
o = Agent(k=0).run({"case_id": "t", "customer_msg": "my bag", "context": []})
check(not o["needs_human"] and not o["low_confidence"], "with no threshold set, confidence doesn't change the decision")

# ------------------------------------------------------------------ real local profile, no Ollama
llm.clear_overrides()
llm.set_mode("local")
ok, why = llm.llm_available("generator")
if not ok:
    o = Agent(k=0).run({"case_id": "t", "customer_msg": "hello", "context": []})
    check(o["needs_human"] and o["reply"] == "" and not o["valid"] and o["model"].startswith("ollama:"),
          f"with no local model the agent fails safe: escalated, no draft ({o['errors'][0][:50]}...)")
llm.set_mode("offline")

# ------------------------------------------------------------------ baselines
print("-- baselines")
for b in (B0(escalate_all=True), B0(escalate_all=False)):
    res = b.run_many(cases)
    check(all(r["intent"] == b.intent and r["reply"] == FIXED_REPLY and r["needs_human"] is b.escalate_all for r in res),
          f"{b.name}: intent '{b.intent}', fixed reply, needs_human={b.escalate_all} for every case")
b1 = B1(retriever=retriever).run_many(cases)
pool_replies = set(retriever.pool.brand_reply)
check(all(r["reply"] in pool_replies for r in b1), "B1 replies are past BA replies, verbatim")
check(all(r["needs_human"] == bool(r["guard"]["hits"]) for r in b1), "B1 escalates exactly when the guard fires")
check(all(all(f in r for f in FIELDS) for r in b0) if (b0 := B0(True).run_many(cases[:2])) else False,
      "baselines share the agent's output shape")

check("openai" not in sys.modules, "the openai package was never imported")
check((CHAT_CACHE.read_bytes() if CHAT_CACHE.exists() else None) == chat_before, "mock outputs never reached the cache")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

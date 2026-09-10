"""The BA support agent (design C4): retrieve similar past cases, make ONE structured call to
the `generator` role, validate what comes back, then apply the deterministic guard.

    from agent import Agent
    Agent().run({"customer_msg": "...", "context": [...]})
    Agent(reply_gate=True)          # "A+gate": a draft that fails the reply checks isn't auto-sent

The provider comes from config/models.json via src/llm.py, so the agent never knows whether
a local model, the optional paid API or a test mock is answering.

Every result has the same shape, whatever happens:
    intent, intent_confidence, needs_human, escalation_reason, reply
plus provenance: the model spec, the retrieved case ids and top retrieval similarity, the
guard's hits and personal-data flags, the model's own escalation decision, which layers
escalated the case (escalated_by), the reply checks that fired, and any validation errors.

Failure is safe by construction. If the provider is unavailable or returns something
unusable, the case is escalated and the error is recorded; nothing is dropped. With a
confidence threshold set, an unsure case goes to a human. With the reply gate on, a draft
that fails the deterministic reply checks (invented specifics, promises, commitments, offers,
too long...) goes to a human with the draft attached. The gate only ever adds escalations.
The agent only sees the customer's message and earlier turns, never BA's actual reply to the
case it is answering.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402
import llm  # noqa: E402
from reply_checks import check_reply, failed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY = ROOT / "data" / "golden" / "taxonomy.json"
REASONS = ROOT / "config" / "escalation_reasons.json"
FAILSAFE = {"intent": None, "intent_confidence": None, "needs_human": True, "escalation_reason": None, "reply": ""}
LOW_CONFIDENCE_REASON = "unclear_request"


def load_schema() -> tuple[list[dict], list[dict]]:
    intents = json.loads(TAXONOMY.read_text(encoding="utf-8"))["intents"]
    reasons = json.loads(REASONS.read_text(encoding="utf-8"))["reasons"]
    return intents, reasons


def system_prompt(intents: list[dict], reasons: list[dict]) -> str:
    ids = ", ".join(i["id"] for i in intents)
    rids = ", ".join(r["id"] for r in reasons)
    return "\n".join([
        "You draft replies for British Airways customer support on Twitter.",
        "Read the customer's message and any earlier turns, then return ONE JSON object:",
        '{"intent": "<intent id>", "intent_confidence": <0 to 1>, "needs_human": <true|false>, '
        '"escalation_reason": "<reason id or null>", "reply": "<draft reply>"}',
        "",
        f"Intents ({ids}): pick the single one that says what the customer wants BA to do.",
        *[f"- {i['id']}: {i['definition']}" for i in intents],
        "",
        "needs_human is true only when any generic public reply would be harmful, risky or make "
        "things worse. A reply that moves the conversation to DM is safe on its own. "
        f"If true, escalation_reason is one of ({rids}); otherwise it is null.",
        *[f"- {r['id']}: {r['definition']}" for r in reasons],
        "",
        "Reply rules:",
        "- At most 280 characters; friendly, concise, British Airways' voice; no sign-off name.",
        "- Never invent specifics: no flight numbers, times, dates, amounts, phone numbers or links "
        "unless the customer gave them.",
        "- Never promise refunds, compensation or outcomes, and never claim to have checked a booking.",
        "- Ask for booking details only by DM, never in public.",
        "- Always write a draft, even when needs_human is true: a human will review it.",
    ])


def user_prompt(case: dict, exemplars: list[dict], pii: list[str]) -> str:
    lines: list[str] = []
    if case.get("context"):
        lines.append("Earlier in the thread:")
        lines += [f"{'British Airways' if t['role'] == 'brand' else 'Customer'}: {t['text']}" for t in case["context"]]
        lines.append("")
    lines += ["Customer message:", case["customer_msg"], ""]
    if pii:
        lines += [f"Note: the customer has posted personal details publicly ({', '.join(pii)}); "
                  "suggest they delete that tweet and continue by DM.", ""]
    if exemplars:
        lines.append("Past British Airways replies to similar messages, for tone and what BA usually offers. "
                     "Do not copy names, numbers or case details:")
        for n, e in enumerate(exemplars, 1):
            lines += [f"{n}. Customer: {e['customer_msg']}", f"   BA: {e['brand_reply']}"]
        lines.append("")
    lines.append("Return the JSON object only.")
    return "\n".join(lines)


def normalise(raw: dict, intent_ids: set[str], reason_ids: set[str]) -> tuple[dict, list[str]]:
    """Coerce a model's JSON into the output schema. Anything unusable is recorded as an error
    and resolved in the safe direction: if the intent, the escalation decision or the reply
    can't be trusted, the case goes to a human. A bad confidence value alone is recorded but
    doesn't escalate; the optional confidence threshold treats it as low confidence."""
    errors: list[str] = []
    unsafe = False
    intent = raw.get("intent")
    if intent not in intent_ids:
        errors.append(f"invalid intent {intent!r}; escalated")
        intent, unsafe = None, True
    try:
        conf = min(max(float(raw.get("intent_confidence")), 0.0), 1.0)
    except (TypeError, ValueError):
        errors.append(f"invalid intent_confidence {raw.get('intent_confidence')!r}")
        conf = None
    nh = raw.get("needs_human")
    if isinstance(nh, str) and nh.strip().lower() in ("true", "false"):
        nh = nh.strip().lower() == "true"
    if not isinstance(nh, bool):
        errors.append(f"invalid needs_human {nh!r}; escalated")
        nh = True
    reason = raw.get("escalation_reason")
    if isinstance(reason, str) and reason.strip().lower() in ("", "null", "none"):
        reason = None
    if reason is not None and reason not in reason_ids:
        errors.append(f"invalid escalation_reason {reason!r}")
        reason = None
    if not nh and reason is not None:
        errors.append("gave a reason but said no human needed; escalated")
        nh = True
    reply = raw.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        errors.append("missing reply; escalated")
        reply, unsafe = "", True
    if unsafe:
        nh = True
    return {"intent": intent, "intent_confidence": conf, "needs_human": nh,
            "escalation_reason": reason if nh else None, "reply": reply.strip()}, errors


class Agent:
    def __init__(self, k: int = 5, use_guard: bool = True, min_confidence: float | None = None,
                 reply_gate: bool = False, retriever=None, temperature: float = 0.0, seed: int = 0,
                 name: str | None = None):
        self.k, self.use_guard, self.min_confidence, self.reply_gate = k, use_guard, min_confidence, reply_gate
        self.temperature, self.seed = temperature, seed
        self.intents, self.reasons = load_schema()
        self.intent_ids = {i["id"] for i in self.intents}
        self.reason_ids = {r["id"] for r in self.reasons}
        self.system = system_prompt(self.intents, self.reasons)
        self._retriever = retriever
        self.name = name or ("A" + ("" if k else "-no-retrieval") + ("" if use_guard else "-no-guard")
                             + (f"-conf{min_confidence}" if min_confidence is not None else "")
                             + ("+gate" if reply_gate else ""))

    @property
    def retriever(self):
        if self._retriever is None:
            from retrieval import Retriever
            self._retriever = Retriever()
        return self._retriever

    def messages(self, case: dict, exemplars: list[dict], pii: list[str]) -> list[dict]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": user_prompt(case, exemplars, pii)}]

    def run(self, case: dict) -> dict:
        return self.run_many([case])[0]

    def run_many(self, cases: list[dict]) -> list[dict]:
        checks = [guard.check(c["customer_msg"], c.get("context")) for c in cases]
        exemplars = (self.retriever.search([c["customer_msg"] for c in cases], self.k)
                     if self.k else [[] for _ in cases])
        reqs = [{"role": "generator", "messages": self.messages(c, e, g["pii"]),
                 "temperature": self.temperature, "seed": self.seed}
                for c, e, g in zip(cases, exemplars, checks)]
        raws = llm.chat_many(reqs)
        spec = llm.spec_for("generator")
        return [self._finish(c, e, g, raw, spec) for c, e, g, raw in zip(cases, exemplars, checks, raws)]

    def _finish(self, case: dict, exemplars: list[dict], g: dict, raw: dict, spec: str) -> dict:
        if "_error" in raw:
            out, errors = dict(FAILSAFE), [raw["_error"]]
        else:
            out, errors = normalise(raw, self.intent_ids, self.reason_ids)
        escalated_by = ["model" if not errors else "invalid_output"] if out["needs_human"] else []
        if self.use_guard:
            final = guard.apply(out["needs_human"], out["escalation_reason"], g["hits"])
        else:
            final = {"needs_human": out["needs_human"], "escalation_reason": out["escalation_reason"],
                     "guard_escalated": False}
        if final["guard_escalated"]:
            escalated_by.append("guard")
        conf = out["intent_confidence"]
        low = self.min_confidence is not None and (conf is None or conf < self.min_confidence)
        if low and not final["needs_human"]:
            final = {**final, "needs_human": True, "escalation_reason": LOW_CONFIDENCE_REASON}
            escalated_by.append("low_confidence")
        fails = failed(check_reply(out["reply"], case["customer_msg"], case.get("context")))
        gated = self.reply_gate and bool(fails) and not final["needs_human"]
        if gated:      # the draft goes to a human; no escalation reason fits a bad draft
            final = {**final, "needs_human": True, "escalation_reason": None}
            escalated_by.append("reply_gate")
        return {"case_id": case.get("case_id"), "system": self.name, "model": spec,
                "intent": out["intent"], "intent_confidence": conf,
                "needs_human": final["needs_human"], "escalation_reason": final["escalation_reason"],
                "reply": out["reply"],
                "llm_needs_human": out["needs_human"], "llm_escalation_reason": out["escalation_reason"],
                "guard": {"hits": g["hits"], "pii": g["pii"], "escalated": final["guard_escalated"]},
                "low_confidence": low,
                "reply_gate": {"on": self.reply_gate, "escalated": gated, "failed_checks": fails},
                "escalated_by": escalated_by,
                "retrieved": [x["case_id"] for x in exemplars],
                "retrieval_top_score": exemplars[0]["score"] if exemplars else None,
                "valid": not errors, "errors": errors}

"""LLM-as-judge interface (reply-quality layer 3 in DESIGN.md section 5).

    from judge import Judge
    j = Judge()             # the "judge" role: locally llama3.2:3b via Ollama (config/models.json)
    j.available()           # (False, "model 'llama3.2:3b' is not pulled ...") until it is
    j.score_many(items)     # items: {"item_id", "customer_msg", "context", "reply"}
    j.pairwise(item, reply_a, reply_b)

Rubric: five dimensions on anchored 1-3 scales plus a binary "sendable as-is", with the
rationale written before the scores. The judge is blind: it sees the customer's message,
earlier turns and one candidate reply, never which system wrote it or BA's actual reply.
Items are sent in a seeded, shuffled order so position in the batch can't track the system,
and results come back in the caller's order. Pairwise comparison runs both orders; a
verdict that flips when the order flips is reported as "inconsistent" (position bias).

A local 3B judge is NOT equivalent to the planned GPT-class judge. Every score records the
judge's model spec, and agreement with the human scores is measured (quadratic-weighted
kappa, metrics.kappa), never assumed.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402

DIMENSIONS = {
    "relevance": ["ignores or misreads what the customer asked", "partly addresses it",
                  "directly addresses what the customer asked"],
    "groundedness": ["states facts, policies, times or amounts the conversation doesn't support",
                     "makes vague claims that might not hold", "claims nothing it can't support"],
    "actionability": ["gives no next step", "gives a vague next step", "gives one clear next step"],
    "tone": ["cold, defensive or wrong for the customer's mood", "acceptable but generic",
             "warm, proportionate and in British Airways' voice"],
    "public_safety": ["unsafe to post: asks for personal data publicly, promises outcomes, or would inflame",
                      "a minor risk", "safe to post publicly"],
}


def rubric_prompt() -> str:
    lines = ["You are grading a draft reply that British Airways might post publicly on Twitter.",
             "Score each dimension 1, 2 or 3 using these anchors:"]
    for dim, anchors in DIMENSIONS.items():
        lines.append(f"- {dim}: " + "; ".join(f"{i} = {a}" for i, a in enumerate(anchors, 1)))
    lines += ["Then decide 'sendable': could BA post this reply as-is, without a human editing it?",
              "Write your rationale first, then the scores. Return ONE JSON object:",
              '{"rationale": "...", ' + ", ".join(f'"{d}": 1-3' for d in DIMENSIONS) + ', "sendable": true|false}']
    return "\n".join(lines)


PAIRWISE_PROMPT = "\n".join([
    "You are comparing two draft replies British Airways might post publicly on Twitter.",
    "Prefer the reply that addresses what the customer asked, claims nothing unsupported, gives a clear",
    "next step, is safe to post publicly, and sounds warm and proportionate. Ignore length on its own.",
    'Return ONE JSON object: {"rationale": "...", "better": "A" | "B" | "tie"}',
])


def _conversation(item: dict) -> list[str]:
    lines = []
    if item.get("context"):
        lines.append("Earlier in the thread:")
        lines += [f"{'British Airways' if t['role'] == 'brand' else 'Customer'}: {t['text']}" for t in item["context"]]
        lines.append("")
    lines += ["Customer message:", item["customer_msg"], ""]
    return lines


def validate(raw: dict) -> tuple[dict, list[str]]:
    errors, out = [], {}
    for d in DIMENSIONS:
        v = raw.get(d)
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = None
        if v not in (1, 2, 3):
            errors.append(f"{d}: {raw.get(d)!r} is not 1-3")
            v = None
        out[d] = v
    s = raw.get("sendable")
    if isinstance(s, str) and s.strip().lower() in ("true", "false", "yes", "no"):
        s = s.strip().lower() in ("true", "yes")
    if not isinstance(s, bool):
        errors.append(f"sendable: {raw.get('sendable')!r} is not true/false")
        s = None
    out["sendable"] = s
    out["rationale"] = raw.get("rationale") if isinstance(raw.get("rationale"), str) else ""
    return out, errors


class Judge:
    def __init__(self, role: str = "judge", seed: int = 0):
        self.role, self.seed = role, seed
        self.system = rubric_prompt()

    def available(self) -> tuple[bool, str]:
        return llm.llm_available(self.role)

    @property
    def spec(self) -> str:
        return llm.spec_for(self.role)

    def messages(self, item: dict) -> list[dict]:
        user = "\n".join(_conversation(item) + ["Draft reply to grade:", item["reply"] or "(empty)", "",
                                                  "Return the JSON object only."])
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def score_many(self, items: list[dict]) -> list[dict]:
        order = random.Random(self.seed).sample(range(len(items)), len(items))
        raws = llm.chat_many([{"role": self.role, "messages": self.messages(items[i]), "temperature": 0.0,
                               "seed": self.seed} for i in order])
        by_pos = dict(zip(order, raws))
        results = []
        for i, item in enumerate(items):
            raw = by_pos[i]
            if "_error" in raw:
                scores, errors = {d: None for d in DIMENSIONS} | {"sendable": None, "rationale": ""}, [raw["_error"]]
            else:
                scores, errors = validate(raw)
            results.append({"item_id": item.get("item_id"), "judge": self.spec, **scores,
                            "valid": not errors, "errors": errors})
        return results

    def _prefer(self, item: dict, first: str, second: str) -> str | None:
        user = "\n".join(_conversation(item) + ["Reply A:", first or "(empty)", "", "Reply B:", second or "(empty)",
                                                  "", "Return the JSON object only."])
        try:
            raw = llm.chat_json(self.role, [{"role": "system", "content": PAIRWISE_PROMPT},
                                            {"role": "user", "content": user}], temperature=0.0, seed=self.seed)
        except Exception:  # noqa: BLE001 - recorded as no verdict
            return None
        v = str(raw.get("better", "")).strip().upper()
        return v if v in ("A", "B", "TIE") else None

    def pairwise(self, item: dict, reply_a: str, reply_b: str) -> dict:
        """Compare in both orders. 'a'/'b'/'tie' only if both orders agree."""
        v1 = self._prefer(item, reply_a, reply_b)                      # A = reply_a
        v2 = self._prefer(item, reply_b, reply_a)                      # A = reply_b
        m1 = {"A": "a", "B": "b", "TIE": "tie"}.get(v1)
        m2 = {"A": "b", "B": "a", "TIE": "tie"}.get(v2)
        consistent = m1 is not None and m1 == m2
        return {"winner": m1 if consistent else "inconsistent", "consistent": consistent,
                "verdicts": [v1, v2], "judge": self.spec}

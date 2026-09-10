"""Baselines (design C6). Same output shape as the agent, so the evaluator treats every
system alike. Neither baseline uses an LLM.

B0 (trivial): the most common intent, one fixed reply, and an escalation extreme. Two
    variants, B0-escalate-all and B0-auto-all, anchor the two ends of the risk-coverage curve.
B1 (simple): TF-IDF + logistic regression intent (trained on pool silver labels), the nearest
    past BA reply copied verbatim, and escalation from the guard rules alone. It answers the
    obvious objection to the agent: "why not just reuse what BA said last time?"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402
import tfidf_intent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SILVER_POOL = ROOT / "artifacts" / "silver" / "pool.jsonl"
FIXED_REPLY = "We're sorry to hear this. Please send us a DM with a few more details and we'll look into it for you."


def _record(case: dict, system: str, model: str, intent: str, conf: float | None, needs_human: bool,
            reason: str | None, reply: str, g: dict | None = None, retrieved: list | None = None) -> dict:
    return {"case_id": case.get("case_id"), "system": system, "model": model,
            "intent": intent, "intent_confidence": conf, "needs_human": needs_human,
            "escalation_reason": reason, "reply": reply,
            "llm_needs_human": None, "llm_escalation_reason": None,
            "guard": {"hits": g["hits"], "pii": g["pii"], "escalated": needs_human} if g else None,
            "retrieved": retrieved or [], "valid": True, "errors": []}


class B0:
    """Trivial: always the most common intent, always the same reply, always the same decision."""

    def __init__(self, escalate_all: bool):
        shares = pd.read_json(SILVER_POOL, lines=True).silver_intent.value_counts(normalize=True)
        self.intent, self.share = str(shares.index[0]), round(float(shares.iloc[0]), 4)
        self.escalate_all = escalate_all
        self.name = "B0-escalate-all" if escalate_all else "B0-auto-all"

    def run_many(self, cases: list[dict]) -> list[dict]:
        return [_record(c, self.name, "none", self.intent, self.share, self.escalate_all, None, FIXED_REPLY)
                for c in cases]


class B1:
    """Simple: TF-IDF intent, nearest past BA reply verbatim, guard-only escalation."""
    name = "B1"
    model = "tfidf-lr (trained on silver labels)"

    def __init__(self, retriever=None, clf=None):
        self.clf = clf or tfidf_intent.train_on_silver()
        self._retriever = retriever

    @property
    def retriever(self):
        if self._retriever is None:
            from retrieval import Retriever
            self._retriever = Retriever()
        return self._retriever

    def run_many(self, cases: list[dict]) -> list[dict]:
        msgs = [c["customer_msg"] for c in cases]
        preds = self.clf.predict(msgs)
        nearest = self.retriever.search(msgs, k=1)
        out = []
        for c, (intent, p), nn in zip(cases, preds, nearest):
            g = guard.check(c["customer_msg"], c.get("context"))
            final = guard.apply(False, None, g["hits"])
            out.append(_record(c, self.name, self.model, intent, p, final["needs_human"], final["escalation_reason"],
                               nn[0]["brand_reply"], g, [nn[0]["case_id"]]))
        return out

"""Deterministic escalation guard (design C5). It runs after the model and can only ever ADD
an escalation, never remove one (monotone), so no prompt or model output can talk its way
past it.

    import guard
    g = guard.check(customer_msg, context)      # {"hits": [...], "pii": [...]}
    final = guard.apply(llm_needs_human, llm_reason, g["hits"])

Each rule maps to one of the eight reasons in config/escalation_reasons.json and fires on
wording that, under the labelling guidelines, makes any generic reply risky.
- "Unclear request" has no rule: keywords can't recognise it.
- Personal data posted publicly (booking reference, email, phone, card number) is flagged but
  does NOT escalate: under our definition a safe public reply exists (move to DM and suggest
  deleting the tweet), so it's a constraint on the reply, not a reason for review.

Rules scan the customer's message and their earlier turns, never BA's turns. They are
deliberately conservative: a false alarm costs an agent a minute, a miss can cost a public
post. The A-without-guard ablation measures what that costs in coverage.

Known bias: the golden set's targeted stratum was picked with keyword triggers that share
vocabulary with these rules, so guard recall there is inflated. Guard metrics are reported on
the uniform 130 (decision log #21).
"""
from __future__ import annotations

import re

# Highest priority first: when several rules fire, the reported reason is the most serious.
PRIORITY = ["safety_medical", "legal_threat", "needs_booking_access", "compensation_dispute",
            "disputed_policy_claim", "repeated_contact", "reputational_risk"]

_URGENT = re.compile(r"\b(?:today|tonight|tomorrow|this (?:morning|afternoon|evening)|right now|"
                     r"in (?:\d+|an?|two|few|a few) (?:hours?|hrs?|mins?|minutes)|"
                     r"at the (?:airport|gate|check-?in|desk)|boarding (?:now|soon)|about to (?:board|fly|miss)|"
                     r"miss(?:ed|ing)? (?:my |the |our )?(?:connection|connecting flight|flight)|stranded|stuck at)\b")
_BOOKING = re.compile(r"\b(?:booking|booked|ticket|flight|check[ -]?in|boarding pass|seat|reservation|rebook\w*)\b")
_PROBLEM = re.compile(r"\b(?:can'?t|cannot|unable|won'?t let|not letting|help|need|change|rebook\w*|cancel+ed|"
                      r"missed|stranded|stuck|no one|nobody)\b")
_HOW_TO = re.compile(r"\bhow (?:do|can|should|would) (?:i|we)\b")

RULES: list[tuple[str, str, re.Pattern]] = [
    ("safety_medical", "medical_or_safety_words", re.compile(
        r"\b(?:medical|hospital|ambulance|paramedic|injur(?:y|ies|ed)|unwell|pregnan\w*|wheelchair|"
        r"disab(?:led|ility)|allerg(?:y|ic|ies)|epilep\w*|diabet\w*|heart attack|seizure|"
        r"emergency landing|passed away|died|death|funeral|bereave\w*|unsafe|not safe)\b")),
    ("legal_threat", "legal_words", re.compile(
        r"\b(?:lawyers?|solicitors?|legal action|legal advice|small claims|court|caa|ombudsman|regulator|"
        r"trading standards|suing|(?:sue|suing) (?:you|ba|british airways|them|the airline)|"
        r"(?:will|going to|gonna|shall) sue)\b")),
    ("compensation_dispute", "money_claim_words", re.compile(
        r"\b(?:compensation|eu ?261|reimburs\w*|expenses|refund (?:not|never|still)|"
        r"still (?:no|waiting for|not received)(?: my| a| the)? refund|"
        r"(?:rejected|declined|denied|refused) (?:my |our |the )?(?:claim|refund|compensation)|"
        r"claim (?:was |has been )?(?:rejected|declined|denied|refused))\b")),
    ("disputed_policy_claim", "contested_fact_words", re.compile(
        r"\b(?:(?:was|were|been) told|told (?:me|us)|you (?:said|promised|told)|"
        r"(?:staff|agent|crew|colleague|captain|manager|she|he) (?:said|told|promised)|promised (?:me|us)|"
        r"not (?:your|the) policy|against (?:your|the) policy|that'?s not true|lied|lying)\b")),
    ("repeated_contact", "repeat_contact_words", re.compile(
        r"\b(?:still (?:no|nothing|waiting|not|haven'?t|hasn'?t)|(?:second|third|fourth|fifth) time|"
        r"(?:several|multiple|many|countless) (?:times|calls|emails|attempts|messages)|"
        r"(?:\d+|two|three|four|five|six|several) (?:weeks|months)|no (?:reply|response|answer)|"
        r"chas(?:e|ed|ing)|keep (?:calling|asking|chasing)|yet again)\b")),
    ("reputational_risk", "high_visibility_words", re.compile(
        r"\b(?:journalist|reporter|the press|media|bbc|newspaper|tabloid|viral|discriminat\w*|"
        r"racis[tm]\w*|sexis[tm]\w*|homophob\w*|abus(?:e|ive)|assault\w*|harass\w*)\b")),
]

_PII = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|__email__")),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("phone", re.compile(r"(?<!\w)\+?\d[\d\s-]{8,}\d(?!\w)")),
]
_BOOKING_REF = re.compile(r"\b[A-Za-z0-9]{6}\b")
_FLIGHT_NO = re.compile(r"^[A-Za-z]{2}\d{3,4}$")


def _customer_text(customer_msg: str, context: list[dict] | None) -> str:
    turns = [t["text"] for t in (context or []) if t.get("role") == "customer"]
    return "\n".join(turns + [customer_msg])


def _booking_refs(text: str) -> list[str]:
    """6-character codes with at least two letters and one digit that aren't flight numbers."""
    return [w for w in _BOOKING_REF.findall(text)
            if sum(c.isalpha() for c in w) >= 2 and any(c.isdigit() for c in w) and not _FLIGHT_NO.match(w)]


def check(customer_msg: str, context: list[dict] | None = None) -> dict:
    """Rule hits (first match per reason, in priority order) and personal-data flags."""
    raw = _customer_text(customer_msg, context)
    text = raw.lower()
    hits = []
    for reason, rule, pat in RULES:
        if reason == "compensation_dispute" and _HOW_TO.search(text) and not re.search(r"\b(?:still|rejected|refused|denied)\b", text):
            continue    # "how do I claim compensation?" is a process question, not a dispute
        m = pat.search(text)
        if m:
            hits.append({"reason": reason, "rule": rule, "evidence": m.group(0)})
    # Urgency is about now. Time words in earlier turns can be stale ("this morning", written a
    # month ago, made a refund follow-up look urgent in the smoke test), so the urgency itself
    # must be in the current message. The booking and problem words may come from the thread,
    # since a short follow-up often relies on it.
    m = _URGENT.search((customer_msg or "").lower())
    if m and _BOOKING.search(text) and _PROBLEM.search(text):
        hits.append({"reason": "needs_booking_access", "rule": "urgent_booking_problem", "evidence": m.group(0)})
    hits.sort(key=lambda h: PRIORITY.index(h["reason"]))
    pii = [kind for kind, pat in _PII if pat.search(raw)]
    if _booking_refs(raw):
        pii.append("booking_reference")
    return {"hits": hits, "pii": pii}


def apply(llm_needs_human: bool, llm_reason: str | None, hits: list[dict]) -> dict:
    """Combine the model's decision with the guard. Monotone: never turns an escalation off."""
    guard_reasons = [h["reason"] for h in hits]
    needs_human = bool(llm_needs_human) or bool(guard_reasons)
    if llm_needs_human and llm_reason:
        reason = llm_reason               # the model already escalated with a reason: keep it
    elif guard_reasons:
        reason = min(guard_reasons, key=PRIORITY.index)
    else:
        reason = None
    return {"needs_human": needs_human, "escalation_reason": reason,
            "guard_escalated": bool(guard_reasons) and not llm_needs_human}

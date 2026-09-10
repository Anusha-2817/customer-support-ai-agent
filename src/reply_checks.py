"""Deterministic reply checks (reply-quality layer 1 in DESIGN.md section 5). No model is
involved, so they are exact and reproducible, and they catch the failures that matter most
for a public post whatever a judge thinks of the prose.

    from reply_checks import check_reply
    check_reply(reply, customer_msg, context)

- too_long: over 280 characters (one tweet);
- fabricated: flight numbers, times, dates, amounts, durations, phone numbers and links in the
  reply that appear nowhere in the conversation so far;
- promises: committing BA to a refund, compensation, payment or an outcome;
- public_pii_request: asking for a booking reference, email, phone number etc. without
  moving to DM;
- claims_booking_check: saying BA has checked or looked at the booking (no system has access);
- empty: no draft at all (e.g. a fail-safe escalation).

A reply passes when none of these fire. Escalated cases are checked too: a human edits the
draft, but a draft that invents facts wastes their time. Past BA replies copied verbatim
(baseline B1) often fail "fabricated", because their specifics belonged to another case.
"""
from __future__ import annotations

import re

MAX_CHARS = 280
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
SPECIFICS = [
    ("flight_number", re.compile(r"\b[A-Z]{2}\s?\d{2,4}\b")),          # matched on the original case
    ("time", re.compile(r"\b\d{1,2}[:.]\d{2}\s?(?:am|pm)?\b|\b\d{1,2}\s?(?:am|pm)\b", re.I)),
    ("date", re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s{_MONTH}\b|\b{_MONTH}\s\d{{1,2}}(?:st|nd|rd|th)?\b|"
                        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", re.I)),
    ("amount", re.compile(r"[£€$]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:gbp|eur|usd|pounds|euros|dollars|avios|points)\b", re.I)),
    ("duration", re.compile(r"\b\d+\s?(?:working\s)?(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b", re.I)),
    ("phone", re.compile(r"(?<!\w)\+?\d[\d\s-]{7,}\d(?!\w)")),
    ("link", re.compile(r"https?://\S+|<link>|\bwww\.\S+|\b[a-z0-9-]+\.(?:com|co\.uk|org|net)\b", re.I)),
]
PROMISE = re.compile(
    r"\b(?:we(?:'ll| will)|i(?:'ll| will)|you(?:'ll| will))\s+(?:definitely\s+|certainly\s+)?"
    r"(?:refund|compensate|reimburse|pay|cover|upgrade|rebook|"
    r"(?:receive|get)\s+(?:a\s+|your\s+|the\s+|full\s+)?(?:refund|compensation|voucher|reimbursement))\b"
    r"|\bguarantee\w*\b|\bfull refund\b|\bentitled to\b", re.I)
PII_ASK = re.compile(
    r"\b(?:send|share|provide|tweet|post|reply with|give us|let us (?:know|have)|what(?:'s| is))\b[^.?!]{0,40}?"
    r"\b(?:booking (?:ref(?:erence)?|number)|reference(?: number)?|confirmation (?:code|number)|e-?mail(?: address)?|"
    r"phone(?: number)?|mobile(?: number)?|passport|date of birth|home address|address|card (?:number|details)|"
    r"ticket number)\b", re.I)
PRIVATE = re.compile(r"\b(?:dm|dms|direct message|private message|privately|message us)\b", re.I)
BOOKING_CLAIM = re.compile(
    r"\b(?:i|we)(?:'ve| have)\s+(?:just\s+)?(?:checked|looked (?:at|into)|reviewed|found)\s+"
    r"(?:your|the)\s+(?:booking|reservation|record|ticket)\b|\byour booking (?:shows|is showing|says)\b", re.I)


def _norm(text: str) -> str:
    """Lowercase, drop leading zeros in flight numbers wherever they occur (BA0462 == BA462),
    then drop whitespace, so a reply token can be looked up anywhere in the conversation.
    The same function normalises both sides, so the comparison stays consistent."""
    t = re.sub(r"\b([a-z]{2})\s?0*(\d{1,4})\b", r"\1\2", text.lower())
    return re.sub(r"\s+", "", t)


def _conversation(customer_msg: str, context: list[dict] | None) -> str:
    return "\n".join([t["text"] for t in (context or [])] + [customer_msg or ""])


def fabricated_specifics(reply: str, customer_msg: str, context: list[dict] | None = None) -> list[dict]:
    """Specific facts in the reply that the conversation never mentioned."""
    seen = _norm(_conversation(customer_msg, context))
    found = []
    for kind, pat in SPECIFICS:
        for m in pat.finditer(reply):
            tok = m.group(0)
            if _norm(tok) not in seen:
                found.append({"kind": kind, "text": tok})
    return found


def public_pii_request(reply: str) -> bool:
    """Asks for personal data in a sentence that doesn't move the conversation to DM."""
    return any(PII_ASK.search(s) and not PRIVATE.search(s) for s in re.split(r"(?<=[.!?])\s+", reply))


def check_reply(reply: str | None, customer_msg: str, context: list[dict] | None = None) -> dict:
    reply = (reply or "").strip()
    fab = fabricated_specifics(reply, customer_msg, context) if reply else []
    promises = [m.group(0) for m in PROMISE.finditer(reply)]
    out = {
        "chars": len(reply),
        "empty": not reply,
        "too_long": len(reply) > MAX_CHARS,
        "fabricated": fab,
        "promises": promises,
        "public_pii_request": public_pii_request(reply) if reply else False,
        "claims_booking_check": bool(BOOKING_CLAIM.search(reply)),
    }
    out["passes"] = not (out["empty"] or out["too_long"] or fab or promises
                         or out["public_pii_request"] or out["claims_booking_check"])
    return out

"""Escalation guard tests (src/guard.py).

    python tests/test_guard.py

Rule-level positives and negatives, personal-data flags, and the monotone property checked
over all 5,937 eval-universe messages. Reads data read-only, loads no model, writes nothing.
Exit code 1 on any failure.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import guard  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


reasons = {x["id"] for x in json.loads((ROOT / "config/escalation_reasons.json").read_text(encoding="utf-8"))["reasons"]}
check({reason for reason, _, _ in guard.RULES} | {"needs_booking_access"} <= reasons, "every rule maps to a configured reason")

cases = [
    ("My mother is in a wheelchair and nobody came to help her off the plane", "safety_medical"),
    ("I'll be speaking to my solicitor about this", "legal_threat"),
    ("Going to report you to the CAA", "legal_threat"),
    ("My flight leaves in 3 hours and the app says my booking is cancelled, please help", "needs_booking_access"),
    ("Still waiting for my refund after two months", "compensation_dispute"),
    ("Your claim team rejected my claim for the cancelled flight", "compensation_dispute"),
    ("The gate agent told me I could bring two bags, now I'm charged", "disputed_policy_claim"),
    ("This is the third time I've asked, still no response", "repeated_contact"),
    ("The crew were openly racist to my family", "reputational_risk"),
]
for text, want in cases:
    hits = guard.check(text)["hits"]
    check(bool(hits) and hits[0]["reason"] == want, f"fires {want}: {text[:45]!r} -> {[h['reason'] for h in hits]}")

quiet = ["Loved the new safety video, so funny!", "Hi Sue, thanks for your help earlier",
         "How do I claim compensation for a delayed flight?", "Can I move next month's flight a day earlier?",
         "Great flight today, thank you to the crew", "What's the baggage allowance to Rome?"]
for text in quiet:
    hits = guard.check(text)["hits"]
    check(not hits, f"stays quiet: {text[:45]!r}" + (f" -> fired {[h['rule'] for h in hits]}" if hits else ""))

check(bool(guard.check("still nothing", [{"role": "customer", "text": "I emailed my solicitor"}])["hits"]),
      "scans the customer's earlier turns")
check(not guard.check("thanks", [{"role": "brand", "text": "We have passed this to our legal team"}])["hits"],
      "ignores BA's own turns")

pii = guard.check("My booking ref is J9VG7T on flight BA0462, email me at a.b@example.com")["pii"]
check("booking_reference" in pii and "email" in pii, f"flags personal data: {pii}")
check(guard.check("Flight BA0462 was lovely")["pii"] == [], "a flight number is not a booking reference")

ev = pd.read_json(ROOT / "data/processed/eval_universe.jsonl", lines=True, convert_dates=False)
mono_ok, fired = True, 0
for text, ctx in zip(ev.customer_msg, ev.context):
    hits = guard.check(text, ctx)["hits"]
    fired += bool(hits)
    if not guard.apply(True, "unclear_request", hits)["needs_human"] or guard.apply(False, None, hits)["needs_human"] != bool(hits):
        mono_ok = False
check(mono_ok, "monotone over all 5,937 eval messages: never removes an escalation, escalates exactly when a rule fires")
print(f"      guard fires on {fired / len(ev):.1%} of eval messages")
check(guard.apply(True, "safety_medical", [{"reason": "legal_threat"}])["escalation_reason"] == "safety_medical",
      "keeps the model's own reason when it already escalated")
check(guard.apply(False, None, [{"reason": "repeated_contact"}, {"reason": "legal_threat"}])["escalation_reason"] == "legal_threat",
      "reports the most serious reason when several rules fire")

print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

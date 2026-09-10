"""Deterministic reply-check tests (src/reply_checks.py). No data files, no model.

    python tests/test_reply_checks.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reply_checks import check_reply  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def kinds(r: dict) -> set[str]:
    return {f["kind"] for f in r["fabricated"]}


ask = "When does my flight leave? Nobody has told us anything"
r = check_reply("Hi, flight BA123 now departs at 10:30 on 12th Dec, sorry for the wait.", ask)
check({"flight_number", "time", "date"} <= kinds(r) and not r["passes"], f"invented flight number, time and date: {kinds(r)}")
r = check_reply("Hi, BA123 is showing a delay; please check the app for updates.", "Is BA123 delayed?")
check("flight_number" not in kinds(r), "a flight number the customer gave is not invented")
r = check_reply("So sorry, BA462 is delayed.", "My flight BA0462 is late")
check(not r["fabricated"], "BA462 matches BA0462")
r = check_reply("Sorry about BA77, please DM us.", "hi", [{"role": "brand", "text": "Which flight? BA77?"}])
check(not r["fabricated"], "specifics mentioned earlier in the thread are not invented")
r = check_reply("You'll get a £50 voucher within 5 working days, or call 0344 493 0787.", "I want compensation")
check({"amount", "duration", "phone"} <= kinds(r), f"invented amount, duration and phone number: {kinds(r)}")
r = check_reply("Please see the details here: <link>", "what's the allowance?")
check("link" in kinds(r), "a link the customer never gave counts as invented (catches verbatim BA replies)")

for text in ["We'll refund you in full.", "You will receive compensation for this.", "We guarantee it won't happen again.",
             "You're entitled to a refund.", "I'll rebook you onto the next flight."]:
    check(bool(check_reply(text, ask)["promises"]), f"promise: {text!r}")
for text in ["We'll look into this for you.", "Please DM us and we'll check what we can do.",
             "We're sorry to hear this, please DM us your booking reference."]:
    check(not check_reply(text, ask)["promises"], f"not a promise: {text!r}")

check(check_reply("Please tweet us your booking reference and email.", ask)["public_pii_request"],
      "asking for a booking reference in public is flagged")
check(check_reply("What is your booking reference?", ask)["public_pii_request"], "a public question for the reference is flagged")
check(not check_reply("Please DM us your booking reference and we'll take a look.", ask)["public_pii_request"],
      "asking by DM is fine")
check(check_reply("I've checked your booking and your seat is confirmed.", ask)["claims_booking_check"],
      "claiming to have checked the booking is flagged")

check(not check_reply("x" * 280, ask)["too_long"] and check_reply("x" * 281, ask)["too_long"], "280 characters fit, 281 don't")
e = check_reply("", ask)
check(e["empty"] and not e["passes"], "an empty draft does not pass")
ok = check_reply("We're sorry to hear this. Please DM us and we'll look into it for you.", ask)
check(ok["passes"], f"a clean reply passes: {ok}")

print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

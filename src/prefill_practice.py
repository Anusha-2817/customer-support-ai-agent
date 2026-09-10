"""Pre-set values for the golden cases the labeller already saw in the practice round.

    python src/prefill_practice.py

Reads data/golden/practice_round_2026-09-10.jsonl (the raw practice saves, kept verbatim)
and writes data/golden/prefill_from_practice.jsonl, one record per case. The labelling tool
uses it to pre-set fields for the labeller to review. Nothing here is a label: a case only
gets a golden label when the labeller presses Enter on it.

Rules agreed with the labeller (decision log #20):
- only the latest save per case counts (6 of 57 cases were saved twice);
- escalation decision, reason, BA-reply rating and note are pre-set from that save;
- the intent is suggested only where the practice category maps cleanly onto the final
  taxonomy, otherwise left blank; either way the labeller confirms it;
- cases saved before the urgent-only "Needs booking access" rule (13:45) get no pre-set
  escalation, reason or rating: they are judged fresh;
- every one of these cases is marked as previously exposed to BA's actual reply.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden"
PRACTICE = GOLD / "practice_round_2026-09-10.jsonl"
OUT = GOLD / "prefill_from_practice.jsonl"
REASONS = ROOT / "config" / "escalation_reasons.json"
RULE_CUTOFF = "2026-09-10T13:45:00"   # urgent-only booking rule took effect here

# Placeholder practice categories with a clean equivalent in the final taxonomy.
# t_other, t_airport and t_onboard have none and are deliberately left out.
INTENT_MAP = {
    "t_disruption": "flight_disruption",
    "t_baggage": "baggage",
    "t_refund": "refund_claim_followup",
    "t_booking": "booking_change",
    "t_loyalty": "loyalty_avios",
    "t_digital": "website_app",
    "t_praise": "other_non_actionable",
}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = read_jsonl(PRACTICE)
    latest: dict[str, dict] = {}
    for r in recs:                       # the tool appends, so file order is save order
        latest[str(r["case_id"])] = r

    golden = {str(r["case_id"]) for r in read_jsonl(GOLD / "to_label.jsonl")}
    intents = {i["id"] for i in json.loads((GOLD / "taxonomy.json").read_text(encoding="utf-8"))["intents"]}
    reasons = {r["id"] for r in json.loads(REASONS.read_text(encoding="utf-8"))["reasons"]}
    outside = sorted(set(latest) - golden)
    if outside:
        sys.exit(f"practice cases not in the golden set: {outside}")
    if not set(INTENT_MAP.values()) <= intents:
        sys.exit("INTENT_MAP points at an intent that is not in data/golden/taxonomy.json")

    rows, skipped = [], []
    for cid, r in latest.items():
        valid = isinstance(r.get("escalate"), bool) and r.get("ba_reply_ok") in {"yes", "no", "unsure"} \
            and (not r["escalate"] or r.get("reason") in reasons)
        if not valid:
            skipped.append(cid)
            continue
        fresh = r["ts"] < RULE_CUTOFF
        rows.append({
            "case_id": cid,
            "practice_ts": r["ts"],
            "practice_intent": r["intent"],
            "intent": INTENT_MAP.get(r["intent"]),
            "escalate": None if fresh else r["escalate"],
            "reason": None if fresh else (r.get("reason") if r["escalate"] else None),
            "ba_reply_ok": None if fresh else r["ba_reply_ok"],
            "note": r.get("note") or "",
            "rejudge_escalation": fresh,
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(recs)} practice saves -> {len(latest)} unique cases -> {len(rows)} pre-set records "
          f"({len(skipped)} invalid skipped) -> {OUT.relative_to(ROOT)}")
    print(f"  intent suggested: {sum(r['intent'] is not None for r in rows)} | intent left blank: "
          f"{sum(r['intent'] is None for r in rows)} | escalation left for fresh judgement: "
          f"{sum(r['rejudge_escalation'] for r in rows)}")


if __name__ == "__main__":
    main()

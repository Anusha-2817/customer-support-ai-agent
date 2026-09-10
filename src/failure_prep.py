"""Failure-analysis preparation: counts and real examples from the headline set, so the top five
failure modes can be chosen and explained from evidence (report section "Failure analysis").

    python src/failure_prep.py --run artifacts/predictions/full-v1

Reads predictions, case texts and the golden labels, for analysis only: nothing here changes the
system, and golden cases are never used for tuning. Writes artifacts/analysis/failure_prep.md.

Sections (all on the 130 uniform cases unless stated):
1. who escalates: the model, the guard, the reply gate;
2. missed escalations by gold reason, with examples (the safety failure);
3. guard false alarms by rule, with the matched evidence (the coverage cost);
4. intent confusions, top pairs with examples, and where B1 gets right what A gets wrong;
5. reply-check failures among A's auto-sent replies, by type, with examples;
6. invalid model outputs;
7. is the model's own confidence informative?
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reply_checks import check_reply, failed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden"
OUT = ROOT / "artifacts" / "analysis" / "failure_prep.md"
N_EXAMPLES = 4


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clip(s: str, n: int = 200) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


def example(c: dict, a: dict, g: dict, extra: str = "") -> list[str]:
    ctx = " (thread)" if c["context"] else ""
    return [f"- **{c['case_id']}**{ctx}: \"{clip(c['customer_msg'])}\"",
            f"  - gold: intent `{g['intent']}`, {'escalate `' + str(g.get('reason')) + '`' if g['escalate'] else 'auto'}"
            f" | A: intent `{a['intent']}`, {'escalate `' + str(a['escalation_reason']) + '`' if a['needs_human'] else 'auto'}"
            f" (by {', '.join(a.get('escalated_by') or []) or 'none'}){extra}",
            f"  - A's draft: \"{clip(a['reply'], 240)}\""]


def build(run_dir: Path, system: str = "A") -> str:
    cases = {str(c["case_id"]): c for c in read_jsonl(GOLD / "to_label.jsonl")}
    labels = {str(r["case_id"]): r for r in read_jsonl(GOLD / "labels_round1.jsonl")}
    A = {str(r["case_id"]): r for r in read_jsonl(run_dir / f"{system}.jsonl")}
    B1 = {str(r["case_id"]): r for r in read_jsonl(run_dir / "B1.jsonl")}
    ids = [i for i, c in cases.items() if c["stratum"] == "uniform" and i in labels and i in A]
    L = [f"# Failure-analysis preparation: system {system}, {len(ids)} uniform (headline) cases", "",
         "Evidence for choosing the top five failure modes. Golden labels are used for analysis only.", ""]

    # 1. who escalates
    by = Counter(tuple(A[i].get("escalated_by") or ["(auto)"]) for i in ids)
    model_esc = sum(bool(A[i]["llm_needs_human"]) for i in ids)
    L += ["## 1. Who escalates", "",
          f"- The model itself asked for a human on **{model_esc}/{len(ids)}** cases; gold labels escalate "
          f"**{sum(labels[i]['escalate'] for i in ids)}/{len(ids)}**.",
          "- Escalation sources: " + "; ".join(f"{'+'.join(k)}: {v}" for k, v in by.most_common()), ""]

    # 2. missed escalations
    missed = [i for i in ids if labels[i]["escalate"] and not A[i]["needs_human"]]
    L += ["## 2. Missed escalations (gold escalate, A auto-sent)", "",
          f"**{len(missed)}** of {sum(labels[i]['escalate'] for i in ids)} must-escalate cases were auto-sent. By gold reason: "
          + ", ".join(f"`{k}` {v}" for k, v in Counter(labels[i].get("reason") for i in missed).most_common()), ""]
    for reason, _ in Counter(labels[i].get("reason") for i in missed).most_common(3):
        L.append(f"### `{reason}`")
        for i in [i for i in missed if labels[i].get("reason") == reason][:N_EXAMPLES]:
            L += example(cases[i], A[i], labels[i])
        L.append("")

    # 3. guard false alarms
    fa = [i for i in ids if not labels[i]["escalate"] and (A[i].get("guard") or {}).get("escalated")]
    rules = Counter(h["rule"] for i in fa for h in A[i]["guard"]["hits"][:1])
    guard_all = [i for i in ids if (A[i].get("guard") or {}).get("hits")]
    guard_right = sum(labels[i]["escalate"] for i in guard_all)
    L += ["## 3. Guard false alarms (gold auto, escalated by the guard)", "",
          f"The guard fired on {len(guard_all)} headline cases; {guard_right} of them are gold escalations "
          f"(precision {guard_right / len(guard_all):.2f}). **{len(fa)}** false alarms, by first rule: "
          + ", ".join(f"`{k}` {v}" for k, v in rules.most_common()) if guard_all else "The guard never fired.", ""]
    for i in fa[:N_EXAMPLES]:
        ev = "; ".join(f"{h['rule']}: '{h['evidence']}'" for h in A[i]["guard"]["hits"])
        L += example(cases[i], A[i], labels[i], f" | guard evidence: {ev}")
    L.append("")

    # 4. intent confusions
    wrong = [i for i in ids if A[i]["intent"] != labels[i]["intent"]]
    pairs = Counter((labels[i]["intent"], A[i]["intent"]) for i in wrong)
    b1_better = [i for i in wrong if B1.get(i, {}).get("intent") == labels[i]["intent"]]
    L += ["## 4. Intent confusions", "",
          f"A is wrong on **{len(wrong)}/{len(ids)}**; B1 (TF-IDF) is right on **{len(b1_better)}** of those.",
          "Top confusions (gold -> A): " + ", ".join(f"`{g}` -> `{p}` {n}" for (g, p), n in pairs.most_common(6)), ""]
    for (g, p), _ in pairs.most_common(3):
        L.append(f"### gold `{g}` -> A `{p}`")
        for i in [i for i in wrong if (labels[i]["intent"], A[i]["intent"]) == (g, p)][:3]:
            L += example(cases[i], A[i], labels[i], f" | B1: `{B1.get(i, {}).get('intent')}`")
        L.append("")
    L += ["Gold intent distribution vs A's predictions: "
          + ", ".join(f"`{k}` gold {v} / A {sum(A[i]['intent'] == k for i in ids)}"
                      for k, v in Counter(labels[i]["intent"] for i in ids).most_common()), ""]

    # 5. reply-check failures among auto-sent replies
    auto = [i for i in ids if not A[i]["needs_human"]]
    fails = {i: failed(check_reply(A[i]["reply"], cases[i]["customer_msg"], cases[i]["context"])) for i in auto}
    kinds = Counter(k for f in fails.values() for k in f)
    L += ["## 5. Reply-check failures among A's auto-sent replies", "",
          f"{sum(bool(f) for f in fails.values())} of {len(auto)} auto-sent drafts fail at least one check: "
          + ", ".join(f"`{k}` {v}" for k, v in kinds.most_common()), ""]
    for kind, _ in kinds.most_common(4):
        L.append(f"### `{kind}`")
        for i in [i for i, f in fails.items() if kind in f][:3]:
            chk = check_reply(A[i]["reply"], cases[i]["customer_msg"], cases[i]["context"])
            detail = {"fabricated": [f["text"] for f in chk["fabricated"]], "promises": chk["promises"],
                      "commitments": chk["commitments"], "offers": chk["offers"]}.get(kind, "")
            L += example(cases[i], A[i], labels[i], f" | flagged: {detail}" if detail else "")
        L.append("")

    # 6. invalid outputs
    bad = [i for i in ids if not A[i].get("valid", True)]
    L += ["## 6. Invalid model outputs", "", f"{len(bad)} of {len(ids)}:"]
    L += [f"- {i}: {A[i]['errors']}" for i in bad]
    L.append("")

    # 7. confidence
    conf = Counter(A[i]["intent_confidence"] for i in ids)
    L += ["## 7. Is the model's own confidence informative?", "",
          "intent_confidence values: " + ", ".join(f"{k}: {v}" for k, v in conf.most_common(6)),
          f"Accuracy when confidence = 1.0: "
          f"{sum(A[i]['intent'] == labels[i]['intent'] for i in ids if A[i]['intent_confidence'] == 1.0)}"
          f"/{sum(A[i]['intent_confidence'] == 1.0 for i in ids)}", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="failure-analysis preparation")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--system", default="A")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(a.run, a.system), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

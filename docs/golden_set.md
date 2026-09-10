# Golden set: sampling and labelling note

200 hand-labelled cases, all from the evaluation period (after 18 November 2017), so the
retrieval pool never contains the conversation being answered. Files in `data/golden/`:
`to_label.jsonl` (the cases), `sample_uniform.jsonl` and `sample_targeted.jsonl` (the two
draws), `labels_round1.jsonl` (the labels; the last save per case wins),
`prefill_from_practice.jsonl` and `practice_round_2026-09-10.jsonl` (the practice round).

## Sampling (`src/sample_golden.py`)

| Stratum | Cases | How drawn | Used for |
|---|---|---|---|
| Uniform | 130 | Salted hash of case_id over all 5,937 evaluation cases | **Every headline number** |
| Targeted | 70 | Rare intents topped up to 10 by silver label, then 10 long threads, then keyword escalation triggers | Enough positives to measure rare intents and escalation recall; reported separately |

- Picking by hash, not by row position, means a case is in or out on its own merits. An earlier
  row-position draw reshuffled all 130 picks when a cleaning fix moved a single row.
- The stratum is hidden from the labeller.
- The targeted triggers share vocabulary with the agent's guard, so guard recall on the targeted
  stratum is biased upward. Guard metrics are therefore quoted on the uniform 130 only.
- Sampling refuses to run once real labels exist, so the set cannot drift.

## What was labelled

For each case: intent (one of 10), whether it must be escalated and why (one of 8 reasons), whether
BA's actual reply was acceptable, an ambiguity flag and an optional note. The definitions and edge
cases are in [labelling_guidelines.md](labelling_guidelines.md).

| | Uniform 130 | Targeted 70 |
|---|---|---|
| Must escalate | 65 (50%) | 45 (64%) |
| Top escalation reasons | disputed policy claim 16, needs booking access 15, repeated contact 12 | repeated contact 15, safety/medical 9 |
| Legal threat | 0 | 3 |
| Largest intents | other/non-actionable 24, flight disruption 18, refund/claim follow-up 17 | refund/claim follow-up 18, website/app 9, loyalty/Avios 9 |
| Rarest intents | loyalty/Avios 3, contact/DM 4 | contact/DM 2, flight disruption 2 |
| Flagged ambiguous | 5 | 1 |
| BA's actual reply acceptable | 106 yes, 23 no, 1 unsure | 49 yes, 20 no, 1 unsure |
| With earlier turns | 40 | 32 |

Half the uniform sample needs a human. That base rate is itself a finding: even a perfect
classifier could auto-send at most half of BA's Twitter traffic under this escalation policy.

## Protocol

- Guidelines written before labelling, then revised once after a practice round (the escalation
  reasons went from 7 to 8; decision log #16).
- Blind: the tool never shows model output. BA's actual reply is revealed only after intent and
  escalation are set, so it cannot anchor them.
- Fixed random order, one labeller (the author). The 200 final labels were saved between
  2026-09-10 19:52 and 2026-09-11 00:07, a median of 38.5 seconds per case (4.8 hours in total).
- **Practice exposure, disclosed.** 57 of the 130 uniform cases had been seen, with BA's replies,
  in the practice round. Their final labels were pre-set from the latest practice judgements and
  reviewed case by case, and the provenance is recorded per case (`prefilled_from_practice`,
  `prior_exposure_to_ba_reply`, `prefill_fields`, `changed_from_prefill`). Headline numbers stay on
  all 130. A sensitivity row drops the 57, and practice-vs-final agreement is reported as a
  secondary analysis (decision log #20).
- **Planned: intra-annotator agreement.** 30 cases (20 uniform, 10 targeted, picked by hash in
  `src/relabel.py`) re-labelled blind at least 24 hours later, giving Cohen's kappa for intent,
  escalation and BA-reply acceptability. Cases in the blind reply-scoring set are left out, so no
  re-labelled case is one the labeller has just seen a drafted reply to. Run
  `tools/label_server.py --only-ids data/golden/relabel_ids.json --out data/golden/labels_round2.jsonl --seed 8`,
  then `python src/relabel.py agree`. Not yet done.

## Limitations

- One labeller. The escalation policy is the labeller's reading of what BA would want, not BA's.
- 130 uniform cases give wide intervals: about ±0.09 on intent accuracy and ±0.12 on
  must-escalate recall.
- Five intents have fewer than 10 uniform cases, so their per-intent F1 is anecdotal.

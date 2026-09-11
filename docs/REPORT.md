# Can an AI agent answer British Airways' Twitter customers on its own?

*Repository: code, data, cached model outputs, the golden set and the human scores; results
reproduce offline in about 3 minutes (see README). Decision log: docs/DECISIONS.md.*

## Summary

**Not yet, and the evidence says why.** On 130 uniformly sampled, hand-labelled BA tweets, the
agent A (qwen2.5:3b with retrieval and a deterministic guard, all running on a laptop CPU)
auto-sends 75% of cases, but 39% of what it auto-sends should have gone to a human (unsafe
auto-send). The bar I set before seeing any result was 5%. Holding that bar leaves 2–6% of traffic
auto-sent. A TF-IDF baseline, B1, has higher intent accuracy than A (0.54 vs 0.40). The component
that works is the deterministic guard: it doubles must-escalate recall, from 0.22 to 0.42, and
A+gate raises it to 0.58. I rated half of A's drafts sendable as-is (the bar was 85%),
and neither local LLM judge agrees with my scores better than chance, so no judge score counts as
evidence here. The honest deployment today is **draft-for-review**: every reply
goes to a human, pre-drafted, with a stated reason when the system thinks it is risky.

## 1. Framing

**What good means.** A good reply is one BA could post publicly without a human editing it. It
answers what was asked, invents nothing (no fees, times or promises), gives one concrete next
step, sounds like BA, and never asks for personal data in public.

**The costs are lopsided.** A bad auto-sent reply is public, permanent and screenshot-able. An
unnecessary escalation costs an agent about a minute. So the target is **safe coverage**: how much
can be auto-sent while bad posts stay rare, not accuracy.

**Escalate** means a human sees the message before anything is posted. A reply saying "please DM
us" is safe to auto-send, because a human picks the case up in DMs. Every one of the 8 escalation
reasons describes a case where a generic reply would do harm: safety or medical, a legal threat, a
compensation dispute, a disputed fee or policy, repeated contact, reputational risk, an unclear
request, and an *urgent* booking problem (travelling within about 24 hours).

**Pre-registered trust bar** (fixed before any results): must-escalate recall ≥ 0.95,
unsafe auto-send ≤ 5%, and ≥ 85% of auto-sent replies sendable as-is.

## 2. What was built

- **Data.** The Kaggle "Customer Support on Twitter" dataset. British Airways was chosen by a scan
  of all brands for the highest rate of self-contained replies (0.194), because those are the
  replies that can be reused as grounding. Threads were reconstructed, multi-part replies merged,
  and agents' sign-offs stripped. A **temporal split** gives a retrieval pool of 17,809 cases
  before 18 Nov 2017 and an evaluation set of 5,937 cases after it. A random split would let the
  agent retrieve replies to the same incident.
- **Taxonomy.** 10 intents, curated by hand from a MiniLM embedding cluster report rather than
  taken from the clusters directly.
- **Agent (A).** It retrieves the 5 most similar past cases and makes one structured LLM call
  (intent, reply, needs-human, reason). A deterministic **guard** runs after it. The guard can
  only add escalations. It has keyword rules for safety or medical issues, legal threats, money
  claims, disputed fees or policies, repeat contact and high-visibility complaints, plus an
  urgent-booking rule whose urgency must be in the current message. It also flags personal data
  the customer posted publicly. Any invalid output escalates (fail-safe).
  **A+gate** also escalates any draft that fails the deterministic reply checks (§3).
- **Baselines.** B0 is the two extremes, escalate everything and auto-send a generic holding reply.
  B1 is TF-IDF with logistic regression, trained on silver labels, that sends BA's nearest past
  reply verbatim, with the same guard. **Ablations**: A without retrieval, A without the guard.
- **Models**, all free and local: MiniLM embeddings, qwen2.5:3b (generator), llama3.2:3b (judge,
  deliberately another model family). Paid APIs are blocked in code. Every output is cached and
  committed, so the full evaluation replays in about 3 minutes without any model.

## 3. How it was evaluated

- **Golden set, 200 blind hand labels** from the evaluation period: 130 uniform (every headline
  number) and 70 targeted at rare intents and likely escalations, reported separately. The
  labelling tool never shows model output, and BA's reply is revealed only after intent and
  escalation are set. Under the pre-registered escalation policy, 65/130 uniform cases (50%) were
  labelled must-escalate; this is a policy-defined evaluation composition, not an estimate of BA's
  real-world escalation rate. *Disclosure:* 57 uniform cases had been seen in a practice round, and
  their labels were reviewed from practice pre-sets.
  A sensitivity row drops them.
- **Metrics.** Intent accuracy and macro-F1. Must-escalate recall. *Unsafe auto-send*, the share of
  auto-sent cases that needed a human. Coverage. Risk–coverage curves. All with 95% bootstrap
  intervals, and systems compared by paired bootstrap on the same cases.
- **Deterministic reply checks.** Length; specifics (fees, flight numbers, times, links) absent
  from the conversation; promises, commitments and offers; public requests for personal data;
  claims to have checked a booking.
- **Leakage control.** Opening the labels file during prediction is treated as an error; every
  prediction run made 0 attempts to access it. Prompt and rule changes were made on a separate
  30-case development set from the pool, never on golden cases.
- **Judge validation.** An LLM judge scores relevance, groundedness, actionability, tone and public
  safety (anchored 1–3) and whether the reply is sendable as-is. It is checked against my blind scores on
  80 replies mixed across systems (30 for development, 50 reported).

## 4. Results

**Headline, 130 uniform cases** (95% bootstrap intervals):

| System | Intent acc. | Macro-F1 | Must-escalate recall | Unsafe auto-send | Coverage |
|---|---|---|---|---|---|
| B0 escalate-all | 0.09 | 0.02 | 1.00 | n/a | 0.00 |
| B0 auto-all | 0.09 | 0.02 | 0.00 | 0.50 [0.42, 0.58] | 1.00 |
| B1 TF-IDF + nearest reply | **0.54** [0.45, 0.62] | 0.48 | 0.26 [0.16, 0.38] | 0.44 [0.35, 0.54] | 0.83 |
| **A** agent | 0.40 [0.32, 0.48] | 0.42 | 0.42 [0.29, 0.54] | 0.39 [0.29, 0.49] | 0.75 |
| A+gate | 0.40 | 0.42 | **0.58** [0.47, 0.70] | **0.36** [0.25, 0.46] | 0.58 |
| A without retrieval | 0.47 [0.38, 0.55] | **0.52** | 0.31 [0.20, 0.43] | 0.43 [0.34, 0.52] | 0.81 |
| A without guard | 0.40 | 0.42 | 0.22 [0.12, 0.32] | 0.45 [0.35, 0.54] | 0.88 |

**Paired differences** (first minus second, same 130 cases):
- **A − B1.** Intent accuracy −0.14 [−0.25, −0.03]; unsafe auto-send −0.05 [−0.09, −0.02];
  coverage −0.08. A has slightly lower unsafe auto-send and lower intent accuracy.
- **A − A without guard.** Unsafe auto-send −0.06 [−0.10, −0.01]; coverage −0.13. The guard earns
  its keep.
- **A − A without retrieval.** Intent accuracy −0.07 [−0.15, +0.01], not significant; unsafe
  auto-send −0.04 [−0.08, −0.00]. Retrieval does not improve intent accuracy. It makes A more
  cautious and teaches BA's style: without retrieval, 18% of auto-sent drafts ask for personal data in
  public, against 10% with it. But retrieval also imports specifics from other conversations
  (invented details in 6% of drafts vs 3%).
- **A+gate − A.** Unsafe auto-send −0.04 [−0.09, +0.01] for 16 points of coverage.

**Risk–coverage.** The model's own confidence cannot rank risk: it is 1.0 on 128 of 130 cases.
With the cheap signals instead (TF-IDF's probability for the agent's intent, top retrieval
similarity), the best operating point under 5% unsafe auto-send is **6% coverage** (A, retrieval
similarity ≥ 0.79) and 4–5% for A+gate. That is the whole auto-send opportunity at today's quality.

**Reply checks on auto-sent drafts.** A 78% pass [70, 86]; B1 87% [80, 93]; A+gate 100% by
construction. A+gate is a final deterministic veto: any draft that fails the deterministic reply
checks is escalated rather than auto-sent, so only the human scores (§5) measure its replies. B1 sends real BA replies, yet 10%
of them carry specifics that don't belong to the conversation: a past reply's flight number or fee
is wrong in a new context. Mimicking BA is not the same as being safe.

**Other subsets.**
- **Targeted 70** (enriched for escalations): A's must-escalate recall is 0.84 [0.72, 0.94], unsafe auto-send
  0.27. This looks much better, but is inflated: the targeting keywords share vocabulary with the
  guard (§7).
- **73 uniform cases never seen in practice**: same ordering as the headline (A intent accuracy 0.44 vs B1
  0.58; A unsafe auto-send 0.37).
- **Practice vs final labels** on the 57 exposed cases: 8 changed a field on review; escalation
  agreement 0.96 (κ 0.92).

**Cost and latency.** Free (local). A: 20 s per case on a Ryzen 5 5500U CPU. Without retrieval:
9 s, because the prompt is shorter.

## 5. Reply quality, and whether the judge can be trusted

**Human scores** on 80 blind replies, mixed across systems (1–3 scales; the tool never shows
which system wrote a reply):

| Replies | n | Sendable as-is | Relevance | Grounded | Actionable | Tone | Public safety |
|---|---|---|---|---|---|---|---|
| A, draft passes the checks | 30 | 0.50 | 2.87 | 2.37 | 2.60 | 2.90 | 2.93 |
| A, draft fails a check | 15 | 0.47 | 3.00 | 2.13 | 2.80 | 2.87 | 2.87 |
| B1, BA's nearest past reply | 30 | 0.37 | 2.23 | 2.23 | 1.73 | 2.40 | 3.00 |
| B0, generic holding reply | 5 | 0.60 | 2.20 | 3.00 | 1.60 | 2.20 | 3.00 |

- **About half of A's drafts are sendable as-is. The bar was 85%.**
- **A's drafts beat recycled BA replies** on relevance (2.9 vs 2.2) and actionability (2.7 vs
  1.7). Sendable as-is: 0.50 vs 0.37, a difference of +0.13 [−0.13, +0.37], which is not significant
  at n = 30. The LLM writes to the actual question; a past reply answers a different one.
- **The deterministic checks barely predict my judgement.** Drafts that fail them are sendable as-is
  47% of the time, against 50% for drafts that pass, and score lower only on groundedness. The checks
  catch real, specific defects, like an invented fee. But most unsendable drafts fail for reasons
  the checks don't look for, so A+gate's "100% pass" says little about quality.
- **The generic holding reply is sendable as-is 3 times in 5, but it is the least actionable.**
  Sendable as-is is a floor, not a goal.
- **The human-scored sample was stratified** to include drafts that passed and failed the
  deterministic checks, so these sendable-as-is rates are not intended as population estimates.
  Drafts that fail a check make up a third of the A items, more than their real share (about a
  fifth), so A is reported per stratum, not pooled.

**The LLM judges do not agree with the human.** Results on the 50 held-out test items:

| Judge | Quadratic κ: relevance / grounded / actionable / tone / public safety | Sendable as-is κ | Judge's sendable as-is rate (human's) |
|---|---|---|---|
| llama3.2:3b (independent) | 0.05 / −0.04 / 0.06 / −0.03 / −0.06 | 0.04 | 0.08 (0.41) |
| qwen2.5:3b (wrote A's replies) | 0.06 / −0.19 / 0.15 / 0.08 / −0.06 | 0.05 | 0.02 (0.42) |

- **Both judges agree with me at chance level on every dimension.** Both are harsher than I am
  (qwen by about a point on a 3-point scale), and they call almost nothing sendable as-is.
- **Self-preference probe.** Relative to my scores, qwen favours A's replies over B1's by 0.26
  points more than llama does. That is the direction the literature predicts (Panickssery et al.,
  2024), but it means little when agreement is at chance.
- **Verbosity probe.** qwen's disagreement with me grows with reply length (r = 0.37); llama's
  doesn't (0.03).
- **Decision: no judge score is used as evidence of reply quality.** The 30 development items were
  not used to rephrase the rubric, because a 3B judge at κ ≈ 0 needs a stronger model, not new
  wording. The judge harness is kept unchanged, to be re-run with a GPT-class judge against the
  same human scores.

**Not measured: the labeller's agreement with themselves.** A blind re-label of 30 cases was
prepared (`src/relabel.py`, `data/golden/relabel_ids.json`), but it needs a gap of at least 24
hours after the first labels, which the submission deadline didn't allow. So there is no measured
ceiling for human consistency, on the labels or on the reply scores.

## 6. Top five failures

Counts are over the 130 headline cases. Every example is a real case.

1. **It misses escalations whose harm depends on context.** 38 of 65 must-escalate cases were
   auto-sent: disputed fee or policy 14, reputational risk 8, repeated contact 5, urgent booking 5,
   compensation 4, safety 2. The model asked for a human on only 16 cases; the guard did most of
   the escalating. *Case 87345:* a passenger on a plane with a damaged door, distressed after 90
   minutes of deliberation, got the auto-sent reply "Please stay in the cabin and follow the crew's
   instructions." *Case 78379:* a customer refusing a £70 fee got "As a premium airline, we strive
   for excellence."
2. **Its confidence carries no information.** Confidence is 1.0 on 128 cases, of which only 50
   were classified correctly. With no usable risk signal, coverage can't be traded for lower
   unsafe auto-send, which is why the 5% operating point sits at 6% coverage.
3. **It collapses intents into "service complaint".** It predicts service complaint 46 times; the
   gold labels contain 12. Of 24 non-actionable tweets (praise, check-ins), it recognises 1
   (F1 0.08, against 0.71 for B1). *Case 2731023:* "Fantastic crew — helpful, always smiling" was
   classified as travel info. Wrong intents matter less for the reply than for routing and
   reporting, but they show the model reads tone, not request.
4. **About 1 auto-sent draft in 5 fails a deterministic check** (21 of 97): public requests for
   personal data 10, invented specifics 6, over length 3, commitments 2. *Case 1647095:* a customer
   charged 100 Canadian dollars for 3 kg of excess weight was told "our standard overweight baggage
   charge is £65". A+gate blocks these, at a cost of 16 points of coverage.
5. **Failures the checks don't catch.** 9 of 97 auto-sent drafts contain a template placeholder
   ("Hi [Customer]…", 6), an invented name ("Thanks for sharing, Sarah", 1), or are written as if
   by the customer (2). *Case 643111:* the auto-sent reply read "Hi, I've been waiting for a
   resolution to my complaint from September… Can you please provide a response?" These are cheap
   to detect, and they are next week's first checks.

## 7. What is misleading about my headline number

1. **The escalation policy is mine, and must-escalate recall and unsafe auto-send depend on it.**
   Half the uniform sample "must" be escalated because I decided, for example, that a disputed fee
   needs a human. BA might auto-send replies to many of those. Under a looser policy, unsafe auto-send would fall without
   the system changing at all.
2. **One labeller, who also designed the taxonomy.** The intents fit my own mental model, which
   flatters intent accuracy for everyone, including B1's silver labels. Neither my agreement with
   myself nor with a second labeller was measured: the prepared re-label round needed a 24-hour
   gap that the deadline didn't allow.
3. **57 of the 130 headline cases were not fully blind.** I saw them, with BA's replies, in a
   practice round. The sensitivity row without them shows the same ordering, but on 73 cases.
4. **Targeted must-escalate recall (0.84) looks like success but isn't.** The targeting keywords share
   vocabulary with the guard, so the guard is tested on cases selected for words it knows.
5. **"100% of A+gate's auto-sent replies pass the checks" is true by construction.** The gate is
   those checks, and my own scores show drafts that fail them are sendable as-is about as often
   as drafts that pass (47% vs 50%).
6. **n = 130.** Intent accuracy is ±0.08 and must-escalate recall ±0.12, and five intents have fewer than 10
   cases.
7. **It is a 3B model on a laptop, on 16 days of late-2017 traffic.** These numbers describe this
   configuration, not what an airline would deploy, and BA's policies have changed since.

## 8. Next week

1. **Escalation as specific questions.** Ask the model one yes/no question per escalation reason
   ("does the customer dispute a fee or policy?") instead of one free "needs human" flag. Develop
   it on the pool development set, then score it once on the golden set.
2. **A risk signal that works.** Self-consistency or a small classifier trained on the pool. The
   aim is a curve with real coverage under 5%, then a per-intent policy (auto-send only where an
   intent's own unsafe auto-send meets the bar with enough cases).
3. **Checks for placeholders, invented names and customer voice** (failure 5), tested on
   paraphrases, not on the golden cases where they were found.
4. **The same harness with a stronger generator and judge** (GPU, or the paid profile, which is
   already wired in behind `--live`). Nothing in the evaluation changes.
5. **A second labeller on 40 cases** for inter-annotator agreement, and a pairwise judge
   against BA's real reply.

*Not done, and stated here rather than hidden:* the similarity-to-BA metric (ROUGE and embedding
similarity) planned in the design, the pairwise judge, the judge position-bias probe, the
30-case blind re-label for intra-annotator agreement (prepared, not run), and inter-annotator
agreement.

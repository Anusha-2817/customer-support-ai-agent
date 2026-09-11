# How the system works, in plain language

This guide follows one customer tweet through the whole system, then explains how the system was
evaluated and what the results mean. Each section names the files involved, so the code can be
read alongside it. The last two sections collect common design questions with short answers, and
where to change things.

---

## 0. The whole story in one paragraph

The data is real tweets that customers sent to British Airways in late 2017, together with BA's real replies. An agent reads a new tweet, decides what it is about (its intent), drafts a reply, and decides whether that reply can be posted automatically or a human must look first (escalate). To test whether it can be trusted, 200 evaluation tweets were hand-labelled under a blind labelling protocol; 57 of the 130 headline cases had prior practice exposure, which is disclosed and tested in a sensitivity analysis. The agent was compared with simpler systems, the rate at which it would post something it shouldn't was measured, and an AI "judge" was checked against human scores. The result: it cannot yet be trusted to post on its own. It is useful for drafting replies that a human approves.

---

## 1. The problem, in everyday terms

- **What a good reply is.** One BA could post publicly without editing. It answers the question,
  invents nothing (no made-up fees or times), gives one next step, sounds like BA, and never asks
  for personal details in public.
- **Why "accuracy" is the wrong goal.** Mistakes don't cost the same. A bad public reply can be
  screenshotted and go viral. An unnecessary escalation costs an agent a minute. So the evaluation
  measures **safe coverage**: how much the agent can handle alone while bad posts stay rare.
- **What "escalate" means.** A human must see the tweet before anything is posted. "Please DM us
  your booking reference" counts as safe to post, because the conversation then moves to private
  messages, where a human takes over.
- **The trust bar**, written down before any results so the goalposts could not move:
  - must-escalate recall of at least 0.95;
  - unsafe auto-send of at most 5%;
  - at least 85% of auto-sent replies sendable as-is.

---

## 2. The data: from a raw spreadsheet to "cases"

**Files:** `src/prepare.py` → `data/processed/pool.jsonl` and `data/processed/eval_universe.jsonl`

The raw file (`twcs.csv`, 516 MB, 2.8 million tweets from many companies) is one row per tweet.
Each row says which tweet it replies to. `prepare.py` turns this into **cases**: one customer
message, the conversation before it, and BA's reply. It does this in four steps:

1. **Thread rebuilding.** The "in reply to" links are followed back to the start, so every case
   carries its earlier turns (the *context*).
2. **Multi-part reply merging.** BA often splits a long answer over several tweets ("1/2", "2/2").
   The parts are joined back together; otherwise half the answers would be cut off.
3. **Cleaning.** User handles become placeholders, links become `<link>`, HTML codes like `&amp;`
   are fixed, and BA agents' sign-offs (for example `^JH`) are stripped. Sign-offs are style, not
   content, and 81% of BA's tweets carried one.
4. **Splitting by time, not at random.** Cases before 18 Nov 2017 form the **retrieval pool**
   (17,809 cases the agent may learn from). Cases after it form the **evaluation universe** (5,937
   cases, used only for testing).

**Why the time split matters.** Twitter support comes in bursts: one cancelled flight produces
hundreds of near-identical tweets within hours. With a random split, the agent could "look up"
BA's reply to the same incident and appear far better than it is. A time split is like studying
last year's exam papers and being tested on this year's.

5,149 near-duplicate customer messages were also removed, so that no single burst dominates.

**Why British Airways.** `scripts/00_brand_scan.py` to `02_pair_yield.py` scan every brand in
the dataset. BA had the highest rate of replies that make sense on their own (19.4%), as opposed
to replies like "DM us" that contain nothing to learn from. More useful past replies means better
material for grounding the agent's answers.

---

## 3. The intent list (taxonomy)

**Files:** `src/taxonomy.py`, `src/silver.py`, `artifacts/taxonomy/`, `data/golden/taxonomy.json`

The intent taxonomy was informed by the data and then curated by hand.

1. **Embeddings.** A small free model (MiniLM) turns every message into a list of 384 numbers.
   Messages with similar meaning get similar lists, so "my bag is lost" and "where is my
   suitcase" end up close together even though they share no words. Closeness is measured by
   **cosine similarity**: 1 means the same direction, 0 means unrelated.
2. **Clustering.** 5,000 message vectors are grouped into clusters (piles of similar messages),
   and examples from each pile are read.
3. **Curation by hand.** **10 intents** were chosen from those piles: flight disruption,
   baggage, booking change, website/app, loyalty/Avios, refund or claim follow-up, contact/DM,
   travel info, service complaint, and other/non-actionable (praise, chit-chat). They map to keys
   1–9 and 0 in the labelling tool.
4. **Silver labels.** Every pool message gets the intent of its nearest cluster. These labels
   are cheap and noisy ("silver", not "gold"), and are used for one thing only: training the
   simple baseline B1.

**The 8 escalation reasons** (`config/escalation_reasons.json`): needs booking access (urgent
only, travelling within about 24 hours), safety or medical, legal threat, compensation dispute,
disputed fee or policy, repeated contact, reputational risk, and unclear request. Each describes
a case where a generic reply would make things worse.

---

## 4. The golden set: the answer key

**Files:** `src/sample_golden.py`, `tools/label_server.py`, `data/golden/`, `docs/golden_set.md`

Testing the agent needs correct answers written by a human: the **golden set**, 200 hand-labelled
cases.

- **130 uniform cases**, picked at random from the evaluation universe. They give fair, unbiased
  numbers, and **every headline number comes from these**.
- **70 targeted cases**, picked on purpose: rare intents, long threads and likely escalations.
  Without them there would be only 3 loyalty tweets to test. They are reported **separately**,
  because mixing them in would distort every rate.
- **"Random" means hash-based.** A case is picked if a scrambled version of its ID (a *hash*)
  ranks high. An earlier version picked by row position, and a cleaning fix that moved one row
  changed all 130 picks. With hashing, each case's inclusion depends only on the case itself.
- **Blind labelling.** The labelling tool never shows the agent's output. BA's real reply appears
  only after the intent and escalation are chosen, so it cannot sway the label.
- **A disclosed weakness.** 57 of the 130 uniform cases were first seen in a practice round, with
  BA's reply visible. Their labels were pre-filled from the practice answers and then reviewed.
  This is recorded per case, and a "sensitivity" result drops those 57 to check that the
  conclusions hold.

---

## 5. The agent (System A), step by step

**Files:** `src/agent.py`, `src/retrieval.py`, `src/guard.py`, `src/reply_checks.py`

One tweet, followed through: *"100 Canadian dollars for 3kg overweight bag… really??? #ba84"*

**Step 1: retrieval (finding similar past cases).** The tweet becomes 384 numbers. It is compared
by cosine similarity with all 17,809 pool cases, and the **top 5** are kept. Each example is a
past customer message plus BA's actual reply. With only 17,809 vectors, a plain numpy
calculation takes milliseconds, so no vector database is needed. The top similarity score is also
kept: when even the best match is weak, the case is unusual.

**Step 2: one structured call to the language model.** qwen2.5:3b, a small model running free
on a laptop CPU through Ollama, receives the rules, the 10 intents, the 8 reasons, the 5 examples,
the conversation and the tweet. It must answer in JSON:

```json
{"intent": "baggage", "intent_confidence": 1.0, "reply": "…", "needs_human": false, "reason": null}
```

One call is used instead of several, because the reply and the escalation decision depend on each
other, and each call takes about 20 seconds on the development laptop (Ryzen 5 5500U).

**Step 3: validation (fail safe).** If the answer is not valid JSON, names an intent that doesn't
exist (the model once wrote "bag"), or has an empty reply, the case is **escalated**. When in
doubt, a human looks. It is never silently auto-sent.

**Step 4: the guard (deterministic rules).** Plain keyword rules run on the customer's text:
safety or medical words, legal words, money-claim words, disputed fees or policies, repeat contact
("still waiting", "third time"), high-visibility words, and an urgent-booking rule whose urgency
must be in the *current* message, not a month-old one. The guard can **only add** escalations,
never remove them (a *monotone* rule), so adding the guard can never make the agent less safe. It
also flags personal data the customer posted publicly.

**Step 5 (A+gate only): the reply gate.** The draft goes through the deterministic **reply
checks** (§8). If it fails any of them, it is escalated instead of posted.

**The output** for each case records the intent, the reply, whether it is escalated, the reason,
and *what* escalated it (`model`, `guard`, `reply_gate` or `invalid_output`). This record is how
the guard was found to do most of the escalating.

In this example, the model auto-sent: "Our standard overweight baggage charge is £65 per bag."
That fee appears nowhere in the conversation; the model invented it. This is failure 4 in the
report.

---

## 6. The comparison systems: why each exists

**Files:** `src/baselines.py`, `src/tfidf_intent.py`, `src/run_systems.py`

A number means little on its own. "39% unsafe auto-send" is only good or bad compared with
something.

| System | What it does | The question it answers |
|---|---|---|
| **B0-escalate-all** | Sends everything to a human | The safe extreme: perfect safety, zero automation |
| **B0-auto-all** | Posts a generic "DM us" reply to everything | The reckless extreme: what does no thinking at all score? |
| **B1** | Word-counting classifier (TF-IDF + logistic regression) for intent, plus BA's most similar past reply posted word for word; same guard | Is the LLM worth it, or would a simple, cheap system do as well? |
| **A-no-retrieval** | A without the 5 examples | Does retrieval help? |
| **A-no-guard** | A without the keyword rules | Does the guard help? |

**TF-IDF in one line:** each message becomes counts of its words, with common words ("the")
down-weighted and distinctive words ("baggage", "refund") up-weighted. Logistic regression then
learns which words point to which intent. It trains in under a second.

---

## 7. How models are called: the `llm.py` layer and the cache

**Files:** `src/llm.py`, `src/providers.py`, `config/models.json`

- **Roles, not brands.** The code never says "call qwen". It asks for a *role* (`embedding`,
  `generator`, `namer` or `judge`), and `config/models.json` says which model plays each role.
  Swapping models means changing that file, not the code.
- **Three modes:**
  - `--local` (the default): free models on the local machine.
  - `--offline`: no model at all. Answers come from the cache, and a missing answer stops the run
    with an error.
  - `--live`: the paid OpenAI API. It refuses to start unless requested explicitly, and the tests
    prove it cannot be reached by accident.
- **The cache.** Every model answer is saved in `cache/chat.jsonl`, keyed by a fingerprint (hash)
  of the model name, the exact input and the settings. A repeated request returns the saved
  answer instantly. Because the cache is committed to git, the whole evaluation re-runs in about
  3 minutes with no model installed, and the results are byte-identical. The model name is part of
  the key, so answers from different models can never be mixed up.

---

## 8. The reply checks: simple rules that catch dangerous replies

**File:** `src/reply_checks.py`

These are pattern-matching rules, not AI:

- **too long**: over 280 characters (the tweet limit).
- **invented specifics** (`fabricated`): money amounts, flight numbers, times, dates, phone numbers
  or links in the reply that appear nowhere in the conversation. A reply that writes the
  customer's "BA0462" as "BA462" is the same flight, not an invention; a test caught a bug here.
- **promises** ("we guarantee"), **commitments** ("our team will be in touch") and **offers**
  ("a voucher"): things an agent can't honour without a human.
- **public request for personal data**: asking for a booking reference or email in a public
  tweet instead of by DM.
- **claims to have checked a booking**: the agent has no access to booking systems, so a claim
  like "I've checked your booking" is always false.

The human scores showed the limit of these checks: drafts that fail them were sendable as-is
about as often as drafts that pass (47% vs 50%). The checks catch specific dangers, not general
quality.

---

## 9. Running the systems, and the "fence"

**Files:** `src/run_systems.py`, `scripts/run_fenced.py`, `artifacts/predictions/`

`run_systems.py` runs each system on the 200 golden cases. It writes one file per system (for
example `artifacts/predictions/full-v1/A.jsonl`) plus `run_info.json`, which records the models,
the mode, the git commit and the timings.

**The fence** (`scripts/run_fenced.py`) proves the agent never saw the answer key. Before the
run, it replaces Python's file-opening function with a version that raises an error if anything
tries to open `labels_round1.jsonl`, and it counts the attempts. Every real run printed "0
attempts". The labels are read only afterwards, by the scorer.

The expensive step (the LLM, 20 s per case) ran once ("live"), filling the cache. All 7 systems
were then replayed offline from the cache into `full-v1`.

---

## 10. Scoring: what each number means

**Files:** `src/evaluate.py`, `src/metrics.py`

On the 130 uniform cases:

- **Intent accuracy**: the share of tweets given the right intent. A 0.40, B1 0.54.
- **Macro-F1**: the average score over the 10 intents, each counted equally, so rare intents
  matter as much as common ones. Plain accuracy can look good by getting only the common intents
  right.
- **Must-escalate recall**: of the tweets that needed a human (65 of 130 under the escalation
  policy), the share the system escalated. For A it is 0.42, so it missed 58% of them.
- **Unsafe auto-send**: of the tweets a system posted automatically, the share that should have
  gone to a human. For A it is 0.39; the bar was 0.05.
- **Coverage**: the share posted automatically. For A it is 0.75. Coverage and unsafe auto-send
  pull in opposite directions: escalating everything gives zero unsafe auto-send and zero coverage.
- **95% confidence interval (bootstrap)**: with only 130 cases, a number could be luck. The 130
  cases are re-drawn *with replacement* 1,000 times, the number is recomputed each time, and the
  middle 95% of results is reported. "0.40 [0.32, 0.48]" means the true value is probably between
  0.32 and 0.48.
- **Paired comparison**: to compare two systems fairly, the *same* re-drawn cases are used for
  both, and the difference is examined. If its interval excludes 0, the difference is real. For
  A vs B1 on intent accuracy it is −0.14 [−0.25, −0.03], so B1 really is better.
- **Risk–coverage curve**: cases are sorted from "safest-looking" to "riskiest-looking" using a
  signal (for example, how similar the best past example was), only the top slice is auto-sent,
  and coverage and unsafe auto-send are tracked as the slice grows. The **best operating point
  under 5%** is the largest slice that stays under the bar. For A that is only 6% of tweets. The
  model's own confidence could not serve as the signal: it was 1.0 on 128 of 130 cases, including
  the wrong ones.

The scorer writes four separate result files:
- headline (130);
- targeted (70);
- sensitivity (the 73 uniform cases never seen in practice);
- practice-vs-final (how often review changed a practice answer: 8 of 57).

---

## 11. The AI judge, and why it had to be checked

**Files:** `src/judge.py`, `src/run_judge.py`, `src/judge_agreement.py`, `tools/score_server.py`

Grading 1,000 replies by hand is slow, so the usual approach is an **LLM-as-judge**: another
model grades each reply on 5 dimensions (relevance, groundedness, actionability, tone, public
safety). Each is scored 1–3 against a written description of what 1, 2 and 3 mean, plus "sendable
as-is: yes/no".

A judge is only useful if it agrees with humans, so it was checked:

- **80 human-scored replies**, mixed from A, B1 and B0, were shown blind and in random order, with
  exactly the judge's rubric wording. 30 were set aside for developing the rubric; the reported
  numbers use the other 50.
- **Kappa (κ)** measures agreement *beyond chance*: 1 is perfect, 0 is no better than guessing.
  "Quadratic" means a near miss (2 vs 3) counts as partial agreement, and a far miss (1 vs 3) as a
  large disagreement.
- **A different model family.** The judge (llama3.2:3b, from Meta) comes from a different family
  than the reply writer (qwen2.5:3b, from Alibaba), because models tend to favour text like their
  own (*self-preference*). To measure this, qwen was also run as a judge and compared.

**Result:** both judges agree with the human scores at chance level (κ between −0.19 and 0.15 on
every dimension). Both are much harsher than the human scorer, and they call almost nothing
sendable as-is. So **no judge score is used as evidence**; reply quality rests on the human
scores. The rubric was deliberately *not* reworded to chase agreement: a 3B model at κ ≈ 0 needs a
stronger model, not new wording, and chasing agreement on 80 items would amount to tuning on the
test.

**The re-label round** (`src/relabel.py`) would measure the labeller's own consistency: 30 cases
re-labelled blind at least 24 hours later and compared with the first labels. No system can be
expected to agree with a labeller more than the labeller agrees with themselves. It is prepared
but **was not run**, because the 24-hour gap conflicted with the submission deadline. The report
lists this as a limitation.

---

## 12. The tests: what they guarantee

**Folder:** `tests/`. All suites run with `for t in tests/test_*.py; do python "$t"; done`.

- Tests use **mock models** (fake models that return fixed answers) and **synthetic labels** in
  temporary folders, so they run in seconds and never touch real data.
- **Fences in the tests:** any attempt to open the real golden labels or the human scores fails
  the test.
- **Cache protection:** a test fails if a mock answer ever lands in the real cache.
- **Paid-call protection:** the tests prove the OpenAI package is never even imported in local runs.
- Unit tests also cover the guard, the reply checks, the metrics (checked against hand-computed
  values), sampling stability, the labelling tools' blindness, and more.

---

## 13. The results on one screen

| | A (agent) | B1 (simple) | What it means |
|---|---|---|---|
| Intent accuracy | 0.40 | **0.54** | The word-counting classifier has higher intent accuracy |
| Must-escalate recall | **0.42** | 0.26 | A escalates more of the tweets that need a human, mostly thanks to the guard |
| Unsafe auto-send | 0.39 | 0.44 | Both far above the 5% bar |
| Human "sendable as-is" rate | **0.49** | 0.37 | A's drafts are more relevant and actionable (not significant at n = 30) |

The ablations:
- **Without the guard**, must-escalate recall falls from 0.42 to 0.22, which makes the guard the
  component that works.
- **Without retrieval**, intent accuracy is slightly higher (not significant), but the agent asks
  for personal data in public nearly twice as often, because the examples teach BA's "please DM
  us" style.

**Conclusion:** the agent is suitable as a **draft writer for human agents**, not as an
auto-responder.

---

## 14. Common design questions

**Why not optimise accuracy?** Errors don't cost the same. A missed escalation is a public
mistake; a false escalation costs a minute. So the main numbers are must-escalate recall and
unsafe auto-send, and the goal is a safe operating point.

**Why split by time?** To prevent leakage. With a random split, the agent could retrieve BA's
reply to the very same incident. The pool strictly precedes the test period.

**Why is the simple baseline better at intent?** Its classifier is trained on 17,809 silver
labels built from the same intent definitions, so it learns each intent's vocabulary. The 3B LLM
sees only the definitions and 5 examples, and it drifts towards "service complaint" whenever a
tweet sounds unhappy (46 predictions against 12 real ones).

**Why keep a keyword guard alongside an LLM?** The model asked for a human on only 16 of 130
tweets, against 65 that needed one. Rules are simple but predictable and testable, and because the
guard can only add escalations, it can never make the system less safe.

**How is tuning on the test set ruled out?** Generation is fenced off the labels (0 attempts,
recorded in each run). Prompt and rule changes were developed on a separate 30-case set from the
pool (`data/dev/dev_cases.jsonl`). The prompt was run as-is on the golden set and not changed
afterwards.

**Why a judge from a different model family?** Models favour their own writing
(self-preference). The effect is measured by also running the reply writer as a judge.

**Does the failed judge undermine the evaluation?** No, it is a finding. The harness validates
the judge before trusting it; the judge failed validation, so it is not used. A stronger judge
can be plugged in through `config/models.json` and re-validated against the same 80 human scores.

**What is misleading about the headline number?**
- The escalation policy is the labeller's own, so must-escalate recall and unsafe auto-send depend
  on that judgement.
- There is one labeller.
- 57 cases were not fully blind.
- The targeted must-escalate recall looks good only because the targeting words overlap with the
  guard's.
- The model is 3B and runs on a laptop.
- The data covers only 16 days of 2017.

**What would another week add?**
- One yes/no question per escalation reason instead of a single vague "needs human?" flag.
- A risk signal that actually ranks risk.
- Checks for the new failure types: placeholders like "[Customer]", invented names, and replies
  written in the customer's voice.
- A re-run with a stronger model and judge.
- A second labeller.

---

## 15. Where to change things

| Change | Where |
|---|---|
| Use a different model for a role | `config/models.json`: edit the `local` profile, e.g. `"generator": "ollama:llama3.2:3b"` |
| Add or change a guard rule | `src/guard.py`, the `RULES` list: `(reason, rule_name, regex)`; add a test in `tests/test_guard.py` |
| Change how many examples are retrieved | `Agent(k=5)` in `src/agent.py` |
| Change the prompt | The prompt text in `src/agent.py` |
| Add a reply check | `src/reply_checks.py`: add a pattern and include it in `failed()`; test in `tests/test_reply_checks.py` |
| Add a paired comparison | `COMPARE` in `src/evaluate.py` |
| Add a system variant | `ALL_SYSTEMS` in `src/run_systems.py` |

**After any change:**
1. Run the tests.
2. Re-run the affected system: `python scripts/run_fenced.py --systems A --local --run-name try-1`.
   New prompts miss the cache, so this calls the local model (about 20 s per case); `--limit 5`
   gives a quick look.
3. Score it: `python src/evaluate.py --run artifacts/predictions/try-1`.

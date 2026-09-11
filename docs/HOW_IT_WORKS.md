# How this project works, in plain language

This is a study guide. It follows one customer tweet through the whole system, then explains how
the system was tested and what the results mean. Each section names the files involved, so you
can open the code next to it. The last two sections list likely interview questions, and where to
change things if you are asked to modify the code live.

---

## 0. The whole story in one paragraph

We took real tweets that customers sent to British Airways in late 2017, together with BA's real
replies. We built an agent that reads a new tweet, decides what it is about (its *intent*),
drafts a reply, and decides whether that reply can be posted automatically or a human must look
first (*escalate*). To find out whether it can be trusted, you hand-labelled 200 tweets it had
never seen. We compared it with simpler systems, measured how often it would post something it
shouldn't, and checked whether an AI "judge" can grade replies the way you do. The answer: it
can't be trusted to post on its own yet. It is useful for drafting replies that a human approves.

---

## 1. The problem, in everyday terms

- **What a good reply is.** One BA could post publicly without editing. It answers the question,
  invents nothing (no made-up fees or times), gives one next step, sounds like BA, and never asks
  for personal details in public.
- **Why "accuracy" is the wrong goal.** Mistakes don't cost the same. A bad public reply can be
  screenshotted and go viral. An unnecessary escalation costs an agent a minute. So we measure
  **safe coverage**: how much the agent can handle alone while bad posts stay rare.
- **What "escalate" means.** A human must see the tweet before anything is posted. "Please DM us
  your booking reference" counts as safe to post, because the conversation then moves to private
  messages, where a human takes over.
- **The trust bar**, written down before any results, so we couldn't move the goalposts:
  - catch at least 95% of the tweets that need a human;
  - at most 5% of auto-posted tweets should have needed a human;
  - at least 85% of auto-posted replies good enough to send as-is.

---

## 2. The data: from a raw spreadsheet to "cases"

**Files:** `src/prepare.py` → `data/processed/pool.jsonl` and `data/processed/eval_universe.jsonl`

The raw file (`twcs.csv`, 516 MB, 2.8 million tweets from many companies) is one row per tweet.
Each row says which tweet it replies to. `prepare.py` turns this into **cases**: one customer
message, the conversation before it, and BA's reply. Four steps:

1. **Rebuild threads.** Follow the "in reply to" links back to the start, so every case carries
   its earlier turns (the *context*).
2. **Merge multi-part replies.** BA often splits a long answer over several tweets ("1/2", "2/2").
   We glue them back together, otherwise half the answers would be cut off.
3. **Clean.** Replace user handles with placeholders, turn links into `<link>`, fix HTML codes like
   `&amp;`, and strip BA agents' sign-offs (for example `^JH`). Sign-offs are style, not content,
   and 81% of BA's tweets had one.
4. **Split by time, not at random.** Cases before 18 Nov 2017 become the **retrieval pool**
   (17,809 cases the agent may learn from). Cases after it become the **evaluation universe**
   (5,937 cases, used only for testing).

**Why the time split matters.** Twitter support comes in bursts: one cancelled flight produces
hundreds of near-identical tweets within hours. With a random split, the agent could "look up"
BA's reply to the same incident and look far better than it is. Splitting by time is like giving
a student last year's exam papers to study, then testing them on this year's.

We also removed 5,149 near-duplicate customer messages, so one burst couldn't dominate.

**Why British Airways?** `scripts/00_brand_scan.py` to `02_pair_yield.py` scanned every brand
in the dataset. BA had the highest rate of replies that make sense on their own (19.4%), as
opposed to replies like "DM us" that contain nothing to learn from. More useful past replies means
better material for the agent to ground its answers in.

---

## 3. The intent list (taxonomy)

**Files:** `src/taxonomy.py`, `src/silver.py`, `artifacts/taxonomy/`, `data/golden/taxonomy.json`

We didn't invent the intents from imagination; we let the data suggest them.

1. **Embeddings.** A small free model (MiniLM) turns every message into a list of 384 numbers.
   Messages with similar meaning get similar lists, so "my bag is lost" and "where is my
   suitcase" end up close together even though they share no words. Closeness is measured by
   **cosine similarity**: 1 means the same direction, 0 means unrelated.
2. **Clustering.** We grouped 5,000 message vectors into clusters (piles of similar messages)
   and looked at examples from each pile.
3. **Curation by hand.** From those piles you and I chose **10 intents**: flight disruption,
   baggage, booking change, website/app, loyalty/Avios, refund or claim follow-up, contact/DM,
   travel info, service complaint, and other/non-actionable (praise, chit-chat). Keyboard keys
   1–9 and 0 in the labelling tool.
4. **Silver labels.** Every pool message gets the intent of the nearest cluster. These labels are
   cheap and noisy ("silver", not "gold"), and are used for only one thing: training the simple
   baseline B1.

**The 8 escalation reasons** (`config/escalation_reasons.json`): needs booking access (urgent
only, travelling within about 24 hours), safety or medical, legal threat, compensation dispute,
disputed fee or policy, repeated contact, reputational risk, and unclear request. Each describes
a case where a generic reply would make things worse.

---

## 4. The golden set: the answer key

**Files:** `src/sample_golden.py`, `tools/label_server.py`, `data/golden/`, `docs/golden_set.md`

To test the agent we need correct answers written by a human: the **golden set**, 200 cases you
labelled.

- **130 uniform cases**, picked at random from the evaluation universe. These give fair,
  unbiased numbers, and **every headline number comes from these**.
- **70 targeted cases**, picked on purpose: rare intents, long threads, and likely escalations.
  Without them we'd have only 3 loyalty tweets to test. They are reported **separately**, because
  mixing them in would distort every rate.
- **"Random" means hash-based.** A case is picked if a scrambled version of its ID (a *hash*)
  ranks high. The first version picked by row position, and one cleaning fix that moved one row
  changed all 130 picks. With hashing, each case's fate depends only on itself.
- **Blind labelling.** The tool never shows the agent's output, and BA's real reply appears only
  after you've chosen the intent and escalation, so it can't sway your judgement.
- **Disclosed weakness.** 57 of the 130 uniform cases were first seen in a practice round, with
  BA's reply visible. Their labels were pre-filled from your practice answers and reviewed. We
  record this per case, and a "sensitivity" result drops those 57 to check the conclusions hold.

---

## 5. The agent (System A), step by step

**Files:** `src/agent.py`, `src/retrieval.py`, `src/guard.py`, `src/reply_checks.py`

Follow one tweet: *"100 Canadian dollars for 3kg overweight bag… really??? #ba84"*

**Step 1: retrieval (find similar past cases).** The tweet becomes 384 numbers. We compare it with
all 17,809 pool cases by cosine similarity and take the **top 5**. Each example is a past customer
message plus BA's actual reply. There are only 17,809 vectors, so a plain numpy calculation takes
milliseconds; no vector database is needed. We also keep the top similarity score: if even the
best match is weak, the case is unusual.

**Step 2: one structured call to the language model.** qwen2.5:3b, a small model running free on
your laptop through Ollama, gets the rules, the 10 intents, the 8 reasons, the 5 examples, the
conversation and the tweet. It must answer in JSON:

```json
{"intent": "baggage", "intent_confidence": 1.0, "reply": "…", "needs_human": false, "reason": null}
```

One call instead of several, because the reply and the escalation decision depend on each other,
and each call costs about 20 seconds on this laptop.

**Step 3: validation (fail safe).** If the answer isn't valid JSON, names an intent that doesn't
exist (the model once wrote "bag"), or has an empty reply, the case is **escalated**. When in
doubt, a human looks. It is never silently auto-sent.

**Step 4: the guard (deterministic rules).** Plain keyword rules run on the customer's text:
safety or medical words, legal words, money-claim words, disputed fees or policies, repeat contact
("still waiting", "third time"), high-visibility words, and an urgent-booking rule (the urgency
must be in the *current* message, not a month-old one). The guard can **only add** escalations,
never remove them; this is called *monotone*. So adding the guard can never make the agent less
safe. It also flags personal data the customer posted publicly.

**Step 5 (A+gate only): the reply gate.** The draft goes through the deterministic **reply
checks** (§8). If it fails any, it is escalated instead of posted.

**The output** for each case records the intent, the reply, whether it's escalated, the reason,
and *who* escalated it (`model`, `guard`, `reply_gate` or `invalid_output`), which is how we
learned that the guard does most of the work.

In our example, the model auto-sent: "Our standard overweight baggage charge is £65 per bag."
That fee appears nowhere in the conversation; the model made it up. That is failure 4 in the
report.

---

## 6. The comparison systems: why each exists

**Files:** `src/baselines.py`, `src/tfidf_intent.py`, `src/run_systems.py`

A number means little alone. "39% unsafe" is only bad or good compared with something.

| System | What it does | The question it answers |
|---|---|---|
| **B0-escalate-all** | Sends everything to a human | The safe extreme: perfect safety, zero automation |
| **B0-auto-all** | Posts a generic "DM us" reply to everything | The reckless extreme: what does no thinking at all score? |
| **B1** | Word-counting classifier (TF-IDF + logistic regression) for intent, and BA's most similar past reply posted word for word; same guard | Is the LLM worth it, or would a simple, cheap system do as well? |
| **A-no-retrieval** | A without the 5 examples | Does retrieval help? |
| **A-no-guard** | A without the keyword rules | Does the guard help? |

**TF-IDF in one line:** each message becomes counts of its words, with common words ("the")
down-weighted and distinctive words ("baggage", "refund") up-weighted. Logistic regression then
learns which words point to which intent. It trains in under a second.

---

## 7. How models are called: the `llm.py` layer and the cache

**Files:** `src/llm.py`, `src/providers.py`, `config/models.json`

- **Roles, not brands.** The code never says "call qwen". It asks for a *role*: `embedding`,
  `generator`, `namer` or `judge`. `config/models.json` says which model plays each role. To swap
  models, you change that file, not the code.
- **Three modes:**
  - `--local` (the default): free models on your laptop.
  - `--offline`: no model at all; answers come from the cache, and a missing answer stops the run
    with an error.
  - `--live`: the paid OpenAI API. It refuses to start unless asked explicitly, and the tests prove
    it can't be reached by accident.
- **The cache.** Every model answer is saved in `cache/chat.jsonl`, keyed by a fingerprint (hash)
  of the model name, the exact input and the settings. Ask the same question again and you get
  the saved answer instantly. Because the cache is committed to git, anyone can re-run the whole
  evaluation in about 3 minutes with no model installed, and get byte-identical results. The
  model's name is part of the key, so answers from different models can never be mixed up.

---

## 8. The reply checks: simple rules that catch dangerous replies

**File:** `src/reply_checks.py`

These are pattern-matching rules, not AI:

- **too long**: over 280 characters (the tweet limit);
- **invented specifics** (`fabricated`): money amounts, flight numbers, times, dates, phone numbers
  or links in the reply that don't appear anywhere in the conversation. If the customer says "BA0462"
  and the reply says "BA462", that's the same flight, not an invention; a bug here was caught by a
  test;
- **promises** ("we guarantee"), **commitments** ("our team will be in touch"), **offers**
  ("we'll give you a voucher"): things an agent can't honour without a human;
- **public request for personal data**: asking for a booking reference or email in a public tweet
  instead of by DM;
- **claims to have checked a booking**: the agent has no access to booking systems, so saying "I've
  checked your booking" is always a lie.

Your scores showed the limit of these checks: drafts that fail them were about as sendable as
drafts that pass (47% vs 50%). They catch specific dangers, not general quality.

---

## 9. Running the systems, and the "fence"

**Files:** `src/run_systems.py`, `scripts/run_fenced.py`, `artifacts/predictions/`

`run_systems.py` runs each system on the 200 golden cases and writes one file per system (for
example `artifacts/predictions/full-v1/A.jsonl`) plus `run_info.json`, which records the models,
the mode, the git commit and the timings.

**The fence** (`scripts/run_fenced.py`) is how we *prove* the agent never saw your answers. Before
running, it replaces Python's file-opening function with a version that raises an error if anything
tries to open `labels_round1.jsonl`, and it counts attempts. Every real run printed "0 attempts".
The labels are read only afterwards, by the scorer.

The expensive step (the LLM, 20 s per case) ran once ("live"), filling the cache. All 7 systems
were then replayed offline from the cache into `full-v1`.

---

## 10. Scoring: what each number means

**Files:** `src/evaluate.py`, `src/metrics.py`

Using the 130 uniform cases:

- **Intent accuracy**: the share of tweets with the right intent. A 0.40, B1 0.54.
- **Macro-F1**: the average score over the 10 intents, each counted equally, so rare intents
  matter as much as common ones. Plain accuracy can look good by getting only the common intents
  right.
- **Must-escalate recall**: of the tweets that needed a human (65 of 130), the share the system
  escalated. The safety number. A 0.42 means it missed 58% of them.
- **Unsafe auto-send**: of the tweets the system posted automatically, the share that should have
  gone to a human. A 0.39. The bar was 0.05.
- **Coverage**: the share posted automatically. A 0.75. Coverage and safety pull in opposite
  directions: escalate everything and you're perfectly safe with zero coverage.
- **95% confidence interval** ("bootstrap"): with only 130 cases, a number could be luck. We
  re-draw 130 cases *with replacement* 1,000 times, recompute the number each time, and report the
  middle 95% of results. "0.40 [0.32, 0.48]" means the true value is probably between 0.32 and
  0.48.
- **Paired comparison**: to compare two systems fairly, we re-draw the *same* cases for both and
  look at the difference. If its interval excludes 0, the difference is real. A vs B1 on intent:
  −0.14 [−0.25, −0.03], so B1 really is better.
- **Risk–coverage curve**: sort cases from "safest-looking" to "riskiest-looking" using a signal,
  such as how similar the best past example was, auto-send only the top slice, and see how coverage
  and unsafe auto-send change as the slice grows. The **best operating point under 5%** is the
  biggest slice that stays under the bar. For A that's only 6% of tweets. We couldn't use the
  model's own confidence, because it said 1.0 on 128 of 130 cases, including the wrong ones.

The scorer writes four separate result files: headline (130), targeted (70), sensitivity (the
73 uniform cases never seen in practice), and practice-vs-final (how often your review changed a
practice answer: 8 of 57).

---

## 11. The AI judge and why we had to check it

**Files:** `src/judge.py`, `src/run_judge.py`, `src/judge_agreement.py`, `tools/score_server.py`

Grading 1,000 replies by hand is slow, so the usual approach is an **LLM-as-judge**: another model
grades each reply on 5 things (relevance, groundedness, actionability, tone, public safety), each
1–3 with a written description of what 1, 2 and 3 mean, plus "sendable as-is: yes/no".

But a judge is only useful if it agrees with humans, so we checked:

- **The 80 replies you scored** were mixed from A, B1 and B0, shown blind in random order, using
  exactly the judge's wording. 30 were set aside for developing the rubric; the reported numbers
  use the other 50.
- **Kappa (κ)** measures agreement *beyond chance*. 1 means perfect, 0 means no better than
  guessing. "Quadratic" means a near miss (2 vs 3) counts as partial agreement, and a far miss
  (1 vs 3) counts as a big disagreement.
- **Different model family.** The judge (llama3.2:3b, from Meta) is from a different family than
  the reply writer (qwen2.5:3b, from Alibaba). Models tend to favour text like their own
  (*self-preference*). To measure this, we also ran qwen as a judge and compared.

**Result:** both judges agree with you at chance level (κ between −0.19 and 0.15 everywhere).
Both are much harsher than you, and they call almost nothing sendable. So **no judge score is
used as evidence**; reply quality rests on your scores. We deliberately did *not* reword the
rubric to chase better agreement: a 3B model at κ ≈ 0 needs a stronger model, not new wording,
and chasing agreement on 80 items would be tuning on the test.

**The re-label round** (`src/relabel.py`) would measure how consistent *you* are: 30 cases
re-labelled blind at least 24 hours later, compared with your first labels. No system can be
expected to agree with you more than you agree with yourself. It is prepared but **was not run**,
because the 24-hour gap conflicted with the deadline. The report lists this as a limitation.

---

## 12. The tests: what they guarantee

**Folder:** `tests/`. Run them all with `for t in tests/test_*.py; do python "$t"; done`.

- Tests use **mock models** (fake models that return fixed answers) and **fake labels** in
  temporary folders, so they run in seconds and never touch real data.
- **Fences in the tests:** any attempt to open the real golden labels or your scores fails the
  test.
- **Cache protection:** a test fails if a mock answer ever lands in the real cache.
- **Paid-call protection:** the tests prove the OpenAI package is never even imported in local runs.
- Plus unit tests for the guard, the reply checks, the metrics (checked against hand-computed
  values), sampling stability, the labelling tools' blindness, and more.

---

## 13. The results on one screen

| | A (agent) | B1 (simple) | What it means |
|---|---|---|---|
| Intent accuracy | 0.40 | **0.54** | The simple word-counting model classifies better |
| Must-escalate recall | **0.42** | 0.26 | A catches more dangerous tweets, mostly thanks to the guard |
| Unsafe auto-send | 0.39 | 0.44 | Both far above the 5% bar |
| Your "sendable" rate | **0.49** | 0.37 | A's drafts are more relevant and actionable (not significant at n = 30) |

The ablations:
- **Without the guard**, recall falls from 0.42 to 0.22, so the guard is the part that works.
- **Without retrieval**, intent is slightly better (not significant), but the agent asks for
  personal data in public twice as often, because the examples teach BA's "please DM us" style.

**Conclusion:** use the agent as a **draft writer for human agents**, not an auto-responder.

---

## 14. Questions you will probably be asked

**"Why not measure accuracy?"** Errors don't cost the same. A missed escalation is a public
mistake; a false escalation costs a minute. So the main numbers are must-escalate recall and
unsafe auto-send, and we look for a safe operating point.

**"Why split by time?"** To stop leakage. With a random split, the agent could retrieve BA's reply
to the very same incident. The pool strictly precedes the test period.

**"Why is the simple baseline better at intent?"** Its classifier was trained on 17,809 silver
labels made with the same intent definitions, so it learns the vocabulary of each intent. The 3B
LLM only sees the definitions and 5 examples, and it drifts towards "service complaint" whenever a
tweet sounds unhappy (46 predictions against 12 real ones).

**"Why keep a keyword guard if you have an LLM?"** The model asked for a human on only 16 of 130
tweets, against 65 that needed one. Rules are dumb but predictable and testable, and because the
guard can only add escalations, it can never make the system less safe.

**"How do you know you didn't tune on the test set?"** Generation is fenced off the labels (0
attempts, recorded in each run). Prompt and rule changes were developed on a separate 30-case set
from the pool (`data/dev/dev_cases.jsonl`). The prompt was run as-is on the golden set and not
changed afterwards.

**"Why a judge from a different model family?"** Models favour their own writing
(self-preference). We measured it by also running the reply writer as a judge.

**"Your judge failed. Isn't that bad?"** It's a finding, not a failure of the evaluation. The
harness is built so the judge is validated before it's trusted. It failed validation, so we
don't use it. A stronger judge can be plugged in through `config/models.json` and re-validated
against the same 80 human scores.

**"What's misleading about your headline number?"** The escalation policy is mine, so the safety
numbers depend on my judgement. One labeller. 57 cases not fully blind. The targeted recall looks
good only because the targeting words overlap with the guard's. A 3B model on a laptop. Only 16
days of 2017.

**"What would you do with another week?"** Ask the model one yes/no question per escalation
reason instead of one vague "needs human?" flag; find a risk signal that actually ranks risk; add
checks for the new failure types (placeholders like "[Customer]", invented names, replies written
in the customer's voice); re-run with a stronger model and judge; add a second labeller.

---

## 15. If asked to change something live: where to look

| Change | Where |
|---|---|
| Use a different model for a role | `config/models.json`: edit the `local` profile, e.g. `"generator": "ollama:llama3.2:3b"` |
| Add or change a guard rule | `src/guard.py`, the `RULES` list: `(reason, rule_name, regex)`. Add a test in `tests/test_guard.py` |
| Change how many examples are retrieved | `Agent(k=5)` in `src/agent.py` |
| Change the prompt | The prompt text in `src/agent.py` |
| Add a reply check | `src/reply_checks.py`: add a pattern and include it in `failed()`. Test in `tests/test_reply_checks.py` |
| Add a paired comparison | `COMPARE` in `src/evaluate.py` |
| Add a system variant | `ALL_SYSTEMS` in `src/run_systems.py` |

**After any change:**
1. Run the tests.
2. Re-run the affected system: `python scripts/run_fenced.py --systems A --local --run-name try-1`.
   New prompts miss the cache, so this calls the local model (about 20 s per case). Add
   `--limit 5` for a quick look.
3. Score it: `python src/evaluate.py --run artifacts/predictions/try-1`.

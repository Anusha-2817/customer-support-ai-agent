# Decision log

A running log, written when each decision is made rather than reconstructed at the end.
The final report keeps the 10–15 most consequential. Format: **decision**, then why.

## Data and brand

1. **Chose the brand with a measured scan, not by size.** The largest brands are the
   worst fit for reply drafting: AppleSupport sends 52.5% of replies to DMs, TMobileHelp 82.8%,
   AskPayPal 73.7%. When most replies are "DM us", reply drafting stops being a real task.
   (`scripts/00_brand_scan.py`)

2. **Widened "deflection" after the first metric misranked Amazon.** v1 counted only DM
   mentions and gave AmazonHelp 0.6% deflection, putting it top. Reading samples showed
   Amazon deflects to web forms instead (12.5%). v2 counts DM, web form and phone.
   (`scripts/01_brand_scan_v2.py`)

3. **Picked British Airways.** It has the highest rate of self-contained replies (0.194) among
   brands with more than 25k tweets, 15% deflection, and the longest inbound messages
   (median 133 characters). Escalation is also a real decision for an airline:
   compensation claims, special assistance and safety issues genuinely shouldn't be auto-answered.
   GWRHelp scored higher on self-containment, but most of its inbound traffic is venting
   with little intent structure. (`scripts/02_pair_yield.py`)

4. **Used the HuggingFace mirror of the Kaggle file.** It needs no Kaggle credentials, and the
   byte count matches (516,508,641). Both sources are cited.

5. **Temporal split (75/25), not random.** Twitter support is bursty: one cancellation produces
   hundreds of near-identical tweets within hours. A random split would let the agent retrieve
   BA's reply to the same incident it is being tested on.

6. **Merged multi-part brand replies.** BA splits long answers across tweets. Treating each
   fragment as its own reply truncates about half the answers and makes BA look less helpful than it is.

7. **Dropped near-duplicate customer messages (5,149).** Otherwise burst events dominate both
   the retrieval pool and any random sample.

14. **Strip agent sign-offs from BA's tweets only, driven by a full inventory.** Found while
    testing the labelling tool: a reply ending "^Davina". Two one-style-at-a-time patches each
    exposed another style, so I inventoried every `^`/`*` in all 29,361 raw BA tweets instead:
    81% carry a sign-off, in a dozen shapes (`^Jane` 17,973, `^R` 1,970, `^HP` 1,794,
    `^DaniH` 1,089, `^Alex C`, `^Lisa.`, `^Beth S.`, `^ Barbara`, `^jm`). Retrieved as exemplars,
    these would teach the agent to sign replies as a real employee. In BA tweets `^` is only ever
    a sign-off, so it is stripped wherever it appears, per fragment before multi-part merging.
    Customer tweets are deliberately not touched: 56 of them address an agent by name
    ("Thanks ^Kev"), which is real content. The name is optional too: after the rebuild, 66 pool
    replies still ended in a bare "^". Covered by 20 unit checks. Lesson: measure the whole
    distribution before writing a cleaning rule, not after, and re-measure on the rebuilt data.

## Agent

8. **"Escalate" means a human must review before anything is posted, not "a human will
   eventually be involved".** So *needing booking access* is not an escalation reason on its own:
   a "please DM us your booking reference" reply is safe to auto-send, and a human takes over in DMs.
   The first draft of the reason list included "needs booking access", which contradicted this
   definition. It was replaced by "vulnerable or distressed customer". (Revised in #16.)

9. **One structured LLM call for intent, reply and escalation.** The reply and the escalation
   decision come from the same reading of the message, so they can't contradict each other.
   It costs a third as much as a three-call chain and is one prompt to explain.

10. **The policy guard can only escalate.** Deterministic rules run after the LLM and can turn
    auto into escalate, never the reverse. That gives an auditable safety floor the prompt cannot talk its way around.

## Infrastructure

11. **Every model call goes through a committed disk cache.** It is keyed by hash(model, prompt,
    params); `LLM_OFFLINE=1` turns a cache miss into an error. This is what makes "reproduce
    the headline results in under 15 minutes" true without an API key.

12. **512-dimension embeddings.** text-embedding-3 supports truncated output; a third of the
    default size costs essentially nothing in retrieval quality at this scale and keeps the committed cache small.

## Evaluation

13. **Blind labelling with a staged reveal.** The labelling tool never shows model output, and it
    hides BA's real reply until intent and escalation are set, so neither the system under test
    nor BA's actual handling anchors the label. The uniform/targeted stratum is hidden too.

15. **Golden-set sampling picks by a hash of case_id, and freezes once labelling starts.**
    Re-running the sampler after the sign-off fix shifted one upstream row and changed all 130
    uniform picks, because row-position sampling reshuffles on any change. After labelling, that
    would silently swap out the golden set. Hash-based selection keeps each case's inclusion
    independent of other rows, and the sampler refuses to run once `labels_round1.jsonl` exists.

16. **Escalation reasons revised to eight, after practice labelling.** Practice labelling
    surfaced a case none of the reasons fit, so "Disputed policy / factual claim" was added: when a
    customer contests a policy or asserts what BA told them, an auto-reply would publicly confirm or
    deny contested facts. "Needs booking access" was restored and "Vulnerable or distressed
    customer" dropped (those cases now fall under Safety / medical or Reputational risk). Restoring
    booking access re-opens the tension in #8, so the labelling guidelines must state when a
    booking-specific case needs review versus a safe DM hand-off. The reasons now live in one file,
    `config/escalation_reasons.json`, read by the labelling tool, the agent and the metrics, so
    their ids cannot drift apart.

    **Resolved:** "Needs booking access" applies only to *urgent* booking cases (travelling within
    about 24 hours, at the airport, about to miss a flight or connection), where a canned "DM us"
    would leave the customer in a queue. Non-urgent booking requests are auto with a DM hand-off.
    A first wording, "escalate when the honest answer depends on the booking", was rejected: it
    contradicts the definition of escalate, because a safe "that depends on your fare, DM us"
    reply always exists, so labels would have been inconsistent. The urgent version fits the
    principle all eight reasons share: a generic reply would be harmful or make things worse.

## Models and providers

17. **Local-first, provider-agnostic models, with paid calls blocked in code.** The OpenAI account
    had no credit and the decision was not to add any, so every model is now reached through a
    role (embedding, generator, namer, judge) mapped in config/models.json. The default local
    profile uses all-MiniLM-L6-v2 embeddings (free, CPU, revision pinned) and Ollama models;
    OpenAI stays as an optional profile. Paid providers refuse to start unless a run passes
    --live. tests/test_llm_modes.py verifies that, plus: the openai package is never imported in
    local runs, --offline computes nothing, and only localhost is contacted. The cost is real: a
    3B local model is weaker than GPT-4-class models and slow on this CPU, and a local judge is
    not equivalent to the planned GPT-class judge. Every result records which model produced it,
    and DESIGN.md section 7 lists the limitations. Without a local LLM, the taxonomy report
    leaves cluster names blank for the human instead of inventing them.

## Taxonomy

18. **Ten intents, curated by hand from a cluster report rather than taken from the clusters.**
    MiniLM embeddings of 5,000 pool messages separate weakly (silhouette 0.018-0.026 for every k
    from 8 to 20), so the k=12 clusters were a reading aid, not the answer: each was read through
    31 real messages plus a keyword profile. Four curation calls: praise stays inside Other, as the
    design planned; seats and upgrades go inside Booking; refunds and chasing an existing case are
    one intent, because that cluster mixes them constantly and a split would create a boundary
    the labeller disagrees with themself on; and the mixed travel-chatter cluster maps to Travel
    information, its only source of silver labels. Two findings from the review: travel-information
    questions had no cluster of their own, and the keyword profile's "safety" hits were posts about
    BA's new safety video, which supports keeping safety as an escalation reason, not an intent.

19. **Silver labels from the curated cluster mapping, and a targeted stratum weighted toward
    escalations.** With no LLM labeller, each pool and eval message gets the intent of its nearest
    taxonomy centroid under the human-curated mapping (src/silver.py). Silver labels only train
    the TF-IDF baseline and steer targeted sampling; they are never used for evaluation. Known
    flaws: every member of a mixed cluster gets one intent, and an intent without its own cluster
    gets none. In the targeted 70, the first version passed unused rare-intent slots (24 of 40) to
    random top-ups; they now go to keyword-triggered likely escalations, raising trigger-matching
    cases in the golden set from 8 to 42, because must-escalate recall is the headline safety
    metric and needs positive cases to measure. The triggers are deliberately separate from the
    agent's guard rules, and headline numbers come from the uniform 130 only.

## Golden set

20. **Reuse the practice-round judgements as reviewed pre-sets, and disclose the exposure.** The
    practice labeller ran on the real uniform sample, so the labeller saw 57 golden cases, with BA's
    replies, before real labelling; practice material should have come from outside the golden set.
    Of three options (label everything fresh, reuse, or swap in unseen cases), the labeller chose to
    reuse. The latest practice save per case (63 saves over 57 cases; 6 were revisits) pre-sets the
    escalation decision, reason, BA rating and note. The intent is suggested only where the placeholder
    category maps cleanly (45 cases) and left blank otherwise (12). The 4 cases saved before the
    urgent-booking rule are judged fresh. Nothing is saved until the labeller reviews the case, and the
    server records on every saved label whether it was pre-set, that BA's reply had been seen, and which
    pre-set fields changed. The 130 uniform cases remain the headline set and the 130/70 split is
    unchanged; practice-vs-final agreement is reported only as a secondary analysis.

## Agent

21. **A deterministic guard that can only add escalations, with personal data as a reply
    constraint rather than a reason to escalate.** Rules for seven of the eight reasons scan the
    customer's message and earlier customer turns, never BA's; Unclear request has no rule because
    keywords can't recognise it. The rules are conservative, since a false alarm costs a minute and
    a miss can cost a public post, and fire on 16.3% of eval messages. Tests include near-misses
    that must stay quiet ("safety video", "Hi Sue", "how do I claim compensation?") and the monotone
    property over all 5,937 eval messages. Personal data posted publicly is flagged but not
    escalated: under our definition a safe reply exists (move to DM, suggest deleting the tweet),
    so the agent's prompt is told instead. Known bias: the targeted golden stratum was sampled with
    keyword triggers that share vocabulary with these rules, so guard recall is reported on the
    uniform 130 only. The sampler's docstring had claimed the two were independent and was corrected.

22. **The agent fails safe, and its confidence threshold is an evaluation choice, not a guess.**
    One structured call returns intent, confidence, escalation and reply; the output is validated
    against the taxonomy and reason list, and anything that can't be trusted goes to a human: no
    JSON, a missing reply, or an intent outside the taxonomy. The last one was a real bug caught by
    the tests: the first version recorded the bad intent but still auto-sent the reply. A bad
    confidence value alone is recorded but doesn't escalate. The design's "low confidence ->
    escalate" branch is an optional threshold, off by default, whose value will be chosen from
    the risk-coverage curve rather than set by hand. The agent's prompt carries the customer's
    message and earlier turns only, never BA's actual reply to the case, which the tests check.

## Evaluation

23. **An evaluation harness built to be hard to fool, including by its author.** Headline numbers
    come from the 130 uniform cases only; the targeted 70 are scored separately, and a sensitivity
    row drops the 57 practice-exposed cases without ever replacing the headline. Every rate carries
    a 95% bootstrap interval, systems are compared by paired bootstrap on the same cases, and a
    statistic undefined in most resamples (too few positive cases) keeps its estimate but gets no
    interval rather than a falsely precise one. The evaluator refuses to report mock-model output
    unless run as a smoke test; the runner refuses to run the agent without its model, rather than
    write 200 fail-safe escalations that look like results; and an offline cache miss aborts a run.
    Reply checks are also reported over auto-sent replies alone, because those are the ones that
    reach customers. The end-to-end test makes opening the real labels file an error, proving the
    harness doesn't read them before they are final. Writing the tests caught one real bug: the
    flight-number normaliser only worked at the start of a string, so a reply repeating the
    customer's own BA0462 as BA462 was flagged as invented.

24. **Four changes from a 5-case smoke test with the real local model, and a rule that golden
    cases are never used for tuning.** The first real run (qwen2.5:3b, 5 uniform golden cases,
    about 33 s per case) returned valid JSON every time but never escalated on its own, reported
    confidence 1.0 on every case, invented a GBP 15 fee in a reply that would have been auto-sent,
    volunteered a voucher, and ran past 280 characters. (1) The guard's urgency rule now reads only
    the current message: "this morning" in a month-old turn had labelled a refund follow-up as an
    urgent booking problem. The guard now fires on 15.1% of eval messages, down from 16.3%.
    (2) A+gate, a new system run alongside an unchanged plain A, stops a draft that fails the
    deterministic reply checks from being auto-sent. Its auto-sent replies pass those checks by
    construction, which results.md states, so judge and human scores measure its replies.
    (3) The reply checks gained commitments ("will be in touch") and offers (vouchers, goodwill
    gestures); on 2,000 real BA replies from the pool they fire 0.9% and 0.4% of the time.
    (4) Risk-coverage uses cheap signals, TF-IDF's probability for the agent's intent and the top
    retrieval similarity, because the model's confidence was uninformative and self-consistency
    would multiply a ~1 h 50 min run. The failure categories came from looking at 5 golden cases,
    so to avoid tuning on the test set the new patterns are generic and tested on paraphrases,
    and any further prompt or rule tuning uses a 30-case development set drawn from the retrieval
    pool (3 per silver intent), whose runs retrieve from the pool minus those cases.

# Decision log: the 15 main decisions

Each entry gives the **decision**, then why, with evidence where there is some. The full running
log, written as decisions were made (27 entries), is in [DECISIONS_full.md](DECISIONS_full.md);
the numbers in brackets point to it.

1. **Chose the brand with a measured scan, not by size: British Airways.** The largest brands send
   most replies to DMs (AppleSupport 52.5%, TMobileHelp 82.8%), which leaves nothing to draft. BA
   has the highest rate of self-contained replies (0.194) among brands with over 25k tweets. Its
   escalations are real decisions too: compensation, safety, special assistance. The first metric
   misranked Amazon, and reading samples caught it. [1–3]

2. **Split by time, not at random, and drop near-duplicates.** Twitter support is bursty: one
   cancellation produces hundreds of near-identical tweets. A random split would let the agent
   retrieve BA's reply to the very incident it is tested on. The pool (17,809) precedes 18 Nov
   2017 and the evaluation set (5,937) follows it; 5,149 near-duplicates were removed. [5, 7]

3. **Clean the data from full inventories, not one-off patches.** Multi-part BA replies are merged,
   otherwise half the answers are truncated. Agent sign-offs were inventoried across all 29,361 BA
   tweets (81% carry one, in a dozen shapes) before any rule was written. Stripping them only from
   BA's tweets keeps the agent from signing replies as a real employee; customer tweets that name
   an agent are left alone. [6, 14]

4. **"Escalate" means a human reviews before anything is posted.** A "please DM us" reply is safe to
   auto-send. Every one of the 8 escalation reasons is a case where a generic reply would do harm,
   so "needs booking access" applies only when the case is urgent (travelling within about 24
   hours). The reasons live in one file, read by the labelling tool, the agent and the metrics.
   [8, 16]

5. **Ten intents curated by hand from a cluster report.** The MiniLM clusters separate weakly
   (silhouette 0.02–0.03 for every k from 8 to 20), so they were a reading aid, not the answer.
   Silver labels from the curated mapping only train the TF-IDF baseline and steer targeted
   sampling; they are never used for evaluation. [18, 19]

6. **Golden set: 130 uniform cases for every headline number, 70 targeted cases reported
   separately.** The targeted cases exist so that rare intents and must-escalate recall can be
   measured at all. Cases are picked by a hash of their ID, because row-position sampling had
   reshuffled all 130 picks after a one-row data fix; the sampler refuses to run once labels
   exist. [15, 19]

7. **Blind labelling with a staged reveal.** The tool never shows model output, and it reveals BA's
   real reply only after intent and escalation are set, so neither the system nor BA's own handling
   can anchor a label. [13]

8. **Practice exposure reused as reviewed pre-sets, and disclosed.** 57 uniform cases had been seen
   in a practice round with BA's replies. Their final labels were pre-set from the practice answers
   and reviewed one by one, with provenance recorded per case. The headline stays on all 130; a
   sensitivity row drops the 57, and practice-vs-final agreement is secondary (8 of 57 changed;
   escalation κ 0.92). [20]

9. **One structured LLM call, and fail safe.** Intent, reply and escalation come from one reading
   of the message, so they can't contradict each other; this costs a third of a three-call chain.
   Invalid JSON, an intent outside the taxonomy, or an empty reply escalates. A test caught the
   first version auto-sending a reply with an invalid intent. [9, 22]

10. **A deterministic guard that can only add escalations.** Keyword rules for seven of the eight
    reasons run after the model and can turn "auto" into "escalate", never the reverse, so they are
    an auditable safety floor. It turned out to be the component that works: without it,
    must-escalate recall falls from 0.42 to 0.22. Its recall is quoted on the uniform set only,
    because the targeted sampling shares its vocabulary. [10, 21]

11. **Local-first, provider-agnostic models, with paid calls blocked in code, and a committed
    cache.** The OpenAI account had no credit, and the decision was not to add any. Roles map to
    models in `config/models.json`: MiniLM, qwen2.5:3b and llama3.2:3b by default, OpenAI only
    behind `--live`. Every output is cached by a hash of (model, input, parameters) and committed,
    so the evaluation replays byte for byte in about 3 minutes with no model. The cost: a 3B model
    on a laptop CPU. [11, 17]

12. **An evaluation built to be hard to fool, including by its author.** The trust bar was fixed
    before any results. Every rate carries a bootstrap interval, and systems are compared by paired
    bootstrap. Generation runs fenced off the labels (0 attempts, recorded per run). The evaluator
    refuses mock-model output, and an offline cache miss aborts a run. [23]

13. **No tuning on golden cases.** A 5-case smoke test led to four general changes: an urgency fix
    in the guard, the A+gate variant alongside an unchanged A, wider reply checks, and cheap risk
    signals. After that, changes were developed on a 30-case set from the retrieval pool. The final
    prompt was run once on the golden set, as is, and its weaknesses are reported, not fixed. [24]

14. **Judge validation against a human, with a mixed blind set and a self-preference probe.** 80
    replies from A, B1 and B0 are scored blind by the human and by the judge with identical rubric
    wording; 30 are for development and 50 are reported. The judge (llama3.2:3b) comes from a
    different model family from the reply writer, and the writer (qwen2.5:3b) is also run as a
    judge to measure self-preference. [25, 26]

15. **No judge score is used as evidence, and the re-label was skipped and disclosed.** Both
    judges agree with the human at chance level (quadratic κ −0.19 to 0.15; sendable κ 0.04 and
    0.05). So reply quality rests on the human scores and the deterministic checks, and the rubric
    was not reworded to chase agreement on 80 items. The 30-case blind re-label needs a 24-hour
    gap that the deadline didn't allow, so it was not run, and the report lists it as a
    limitation. [27]

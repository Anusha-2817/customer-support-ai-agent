# Labelling guidelines: golden set

*Status: draft for review. The escalation section is complete; the intent section is written
after the taxonomy is curated. These guidelines must be final before real labelling starts.
Any change after labelling begins goes in the change log at the bottom, with a date, because
a mid-way change makes earlier labels inconsistent with later ones.*

---

## How to label

Start the tool with `python tools/label_server.py` and open http://localhost:8765.

For each item, in this order:
1. Read the earlier thread turns (if shown) and the customer message.
2. Pick the **intent**.
3. Decide **auto or escalate**, and pick a **reason** if you escalate.
4. BA's actual reply is revealed only now. **Rate it.**
5. Flag as **ambiguous** if appropriate, and add a note.
6. Press **Enter** to save.

| Key | Action |
|---|---|
| `1`…`0`, `-`, `=` | Intent |
| `A` / `E` | Auto / escalate |
| `Shift+1`…`Shift+8` | Escalation reason |
| `Y` / `N` / `U` | BA's reply acceptable: yes / no / unsure |
| `M` | Toggle ambiguous |
| `Enter` | Save and go to next (`Ctrl+Enter` inside the note box) |
| `[` / `]` | Previous / skip |

Stay blind: don't look the case up anywhere else, and never look at model output while labelling.

---

## 1. Intent

*Written after taxonomy curation.* The rule for choosing already applies: pick the **single**
intent that answers **"what does the customer want BA to do?"**, not what the message is about.

---

## 2. Auto or escalate?

The question is: **must a human review before anything is posted?**

- **Auto**: some reply exists that could safely be posted without review. A DM hand-off counts:
  "DM us your booking reference and we'll look into it" is a safe reply.
- **Escalate**: any generic reply would be **harmful, risky, or make things worse**. Every reason
  below is a specific version of that.

Two rules that prevent most disagreements:
- **Label the message, not a reply.** Ask "does a safe reply exist?", not "was BA's reply safe?"
  (That is why BA's reply is hidden at this point.)
- **Use only what's on screen**: the customer message plus the earlier turns shown. If something
  isn't stated, assume it isn't true. For example, if the message doesn't say they're travelling soon,
  treat it as not urgent.

An ordinary angry complaint is usually **auto**: a sincere apology plus a next step is a safe reply.
Escalate it only if one of the reasons below applies.

### Reasons

**1. Needs booking access** (urgent only)
Urgent action is needed on this customer's own booking: travelling within about 24 hours, at the
airport, or about to miss a flight or connection. A generic "DM us" would leave them waiting in a queue.
- *Escalate:* "Flight leaves in 3 hours and the app says my booking is cancelled."
- *Auto:* "Can I move next month's flight a day earlier?" or "Please update my Executive Club details."
  Both get a DM hand-off.

**2. Safety / medical**
Safety, security or medical issue: illness, injury, a condition affecting travel, special
assistance, or a stranded or vulnerable passenger.

**3. Legal threat**
Mentions lawyers, suing, small claims, the CAA or another regulator or ombudsman.
"I'm going to complain" on its own is not a legal threat.

**4. Compensation dispute**
The customer claims or disputes money: EU261 compensation, a refund amount or delay, expenses, a rejected claim.
- *Auto instead:* "How do I claim for a delayed flight?" is a process question with a safe answer.

**5. Disputed policy / factual claim**
The customer contests a policy or asserts what happened or what BA told them. An auto-reply
would publicly confirm or deny contested facts.
- *Auto instead:* "What's your baggage allowance to Rome?" asks what the policy is; it doesn't dispute it.

**6. Repeated contact**
Already raised with BA and still unresolved: "third time", "still no reply", "been waiting weeks",
or earlier turns show BA already responded without resolving it.

**7. Reputational risk**
High visibility or sensitive: media, abuse or a viral thread, discrimination, bereavement, or a
situation where a templated reply would look callous.

**8. Unclear request**
You can't tell what the customer wants (sarcasm, missing context, ambiguity), so guessing in public is risky.
- *Auto instead:* praise, thanks or banter, where a short friendly reply (or none) is clearly fine.

*The examples above are illustrations written for these guidelines, not real tweets.*

### More than one reason applies
Pick the one that most shapes **how the human should respond**, and name the others in the note.

---

## 3. Rating BA's actual reply

"Would this reply be acceptable to post as-is?" Judge it against the definition of a good reply
in `docs/DESIGN.md` §1: it answers what was asked, gives a clear next step, has an appropriate tone,
and asks for no personal data **in public**. Asking for details by DM is fine.
Use **Unsure** when you'd need information you don't have to judge it.

---

## 4. Ambiguous flag

Flag an item when a reasonable second labeller could choose differently on **intent or escalation**,
and say why in the note. Ambiguous items are reported separately, and they show where the
guidelines need sharpening.

---

## Change log
- 2026-09-10: Escalation reasons set to eight (decision log #16). "Needs booking access" limited
  to urgent booking cases.

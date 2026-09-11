# British Airways Twitter support agent

An AI support agent for British Airways' Twitter customer service, built on the public
"Customer Support on Twitter" dataset. For each customer tweet it classifies the intent (10
intents), drafts a reply grounded in BA's own past replies, and decides whether the reply can be
auto-sent or must go to a human (8 escalation reasons). It is evaluated against a hand-labelled
golden set, two baselines and three ablations.

Everything runs on free local models (MiniLM embeddings, qwen2.5:3b as the reply writer,
llama3.2:3b as the judge, via Ollama). Every model output is cached and committed, so the
results replay without any model at all. Paid APIs are blocked in code unless a run is started
with `--live`.

- **Report: [docs/REPORT.md](docs/REPORT.md)**
- Decision log, the 15 main decisions: [docs/DECISIONS.md](docs/DECISIONS.md); the full log is
  in [docs/DECISIONS_full.md](docs/DECISIONS_full.md)
- Plain-language walkthrough of the implementation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)
- Design and methodology: [docs/DESIGN.md](docs/DESIGN.md)
- Labelling guidelines: [docs/labelling_guidelines.md](docs/labelling_guidelines.md)
- Results: [artifacts/predictions/full-v1/results.md](artifacts/predictions/full-v1/results.md)
- Golden set sampling and labelling: [docs/golden_set.md](docs/golden_set.md)

## Headline results (130 uniformly sampled golden cases, 95% bootstrap intervals)

| System | Intent accuracy | Macro-F1 | Must-escalate recall | Unsafe auto-send | Coverage |
|---|---|---|---|---|---|
| B0-escalate-all | 0.09 | 0.02 | 1.00 | n/a | 0.00 |
| B0-auto-all | 0.09 | 0.02 | 0.00 | 0.50 [0.42, 0.58] | 1.00 |
| B1 (TF-IDF + nearest reply) | **0.54** [0.45, 0.62] | 0.48 | 0.26 [0.16, 0.38] | 0.44 [0.35, 0.54] | 0.83 |
| A (agent) | 0.40 [0.32, 0.48] | 0.42 | 0.42 [0.29, 0.54] | 0.39 [0.29, 0.49] | 0.75 |
| A+gate | 0.40 | 0.42 | **0.58** [0.47, 0.70] | **0.36** [0.25, 0.46] | 0.58 |
| A-no-retrieval | 0.47 [0.38, 0.55] | **0.52** | 0.31 [0.20, 0.43] | 0.43 [0.34, 0.52] | 0.81 |
| A-no-guard | 0.40 | 0.42 | 0.22 [0.12, 0.32] | 0.45 [0.35, 0.54] | 0.88 |

*Unsafe auto-send* is the share of auto-sent cases that should have gone to a human; *coverage*
is the share auto-sent. Half of the uniform sample must be escalated, so auto-sending everything
scores 0.50.

- **No system is safe to auto-send.** The best, A+gate, still auto-sends 36% of cases that needed
  a human. Holding unsafe auto-send under 5% leaves 2-6% coverage, so the agent is a
  draft-for-review tool, not an auto-responder.
- **The simple baseline classifies intent better.** A vs B1: -0.14 [-0.25, -0.03] intent
  accuracy. A is safer than B1 (-0.05 [-0.09, -0.02] unsafe auto-send), mostly thanks to the
  deterministic guard: without it, recall falls from 0.42 to 0.22 and unsafe auto-send rises by
  0.06 [0.01, 0.10].
- **Retrieval makes the agent more cautious, not more accurate.** Without retrieved examples,
  intent accuracy is 0.07 higher, but the paired interval [-0.01, +0.15] includes zero. Unsafe
  auto-send is 0.04 [0.00, 0.08] higher.
- The model's own confidence is 1.0 on 128 of 130 cases, so it can't rank risk.

## Reproduce the headline results (offline, about 3 minutes, no model needed)

Python 3.11 or newer. On Windows, set `PYTHONIOENCODING=utf-8` (tweets contain emoji).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_fenced.py --systems B0-escalate-all,B0-auto-all,B1,A,A+gate,A-no-retrieval,A-no-guard --offline --run-name repro
python src/evaluate.py --run artifacts/predictions/repro
```

`--offline` replays `cache/` and runs no model; a cache miss stops the run instead of computing
anything. `scripts/run_fenced.py` makes opening the golden labels an error while predictions are
generated, and prints the number of attempts (0). `evaluate.py` then reads the labels, for scoring
only, and writes:

| File | Contents |
|---|---|
| `results_headline_uniform.md` | **Headline**: 130 uniformly sampled cases, 95% bootstrap intervals, paired comparisons, risk-coverage |
| `results_targeted.md` | 70 cases enriched for escalation triggers, reported separately |
| `results_sensitivity_unexposed.md` | Uniform cases the labeller had not seen in a practice round |
| `results_practice_vs_final.md` | The labeller's practice-round judgements vs their final labels |
| `results.md` / `results.json` | All of the above together |

The committed run is `artifacts/predictions/full-v1/`. On the author's laptop (Ryzen 5 5500U, CPU
only), replaying and scoring took 187 seconds and reproduced the committed predictions byte for
byte.

## Systems

| System | What it is |
|---|---|
| B0-escalate-all | Sends everything to a human (the safety ceiling, zero coverage) |
| B0-auto-all | Auto-sends a generic holding reply to everything |
| B1 | TF-IDF + logistic regression intent, BA's nearest past reply sent verbatim, deterministic guard |
| A | Retrieval of 5 similar past cases, one structured LLM call, deterministic guard |
| A+gate | A, but a draft that fails the deterministic reply checks is escalated instead of sent |
| A-no-retrieval | A without retrieved examples (ablation) |
| A-no-guard | A without the deterministic guard (ablation) |

## Re-run with the local models (about 70 minutes per LLM system on a laptop CPU)

1. Install [Ollama](https://ollama.com), then `ollama pull qwen2.5:3b` and `ollama pull llama3.2:3b`.
2. `pip install -r requirements-local.txt`
3. `python scripts/run_fenced.py --systems A --local --run-name my-run` (and the other systems).

`--local` never calls a paid API. New outputs are appended to the cache.

## Judge and human scores

- Blind human scoring of 80 replies (mixed across systems, BA's actual reply never shown):
  `python tools/score_server.py`, then open http://localhost:8766. Scores go to
  `data/human_scores/scores_round1.jsonl`.
- LLM judge on the same 80 replies:
  `python src/run_judge.py --items data/human_scores/items.jsonl --out artifacts/judge/human80 --local`
- Self-preference probe (the reply-writing model as judge):
  `python src/run_judge.py --items data/human_scores/items.jsonl --role generator --out artifacts/judge/human80-selfjudge --local`
- Agreement, once human scores exist:
  `python src/judge_agreement.py --judge artifacts/judge/human80 --selfjudge artifacts/judge/human80-selfjudge`

## Tests

```bash
for t in tests/test_*.py; do python "$t" || echo "FAILED: $t"; done
```

The tests use mock models and synthetic labels in temporary directories. They fail if the real
golden labels or human scores are opened, or if a mock output reaches the cache.

## Data

The repo ships the processed BA subset (`data/processed/`): 17,809 retrieval-pool cases before
18 November 2017 and 5,937 evaluation cases after it. The 516 MB raw file is not included;
`python src/prepare.py` rebuilds the subset from `data/raw/twcs.csv`. Golden cases are drawn
from the evaluation cases only, so retrieval can never see the conversation being answered.

## Repository map

| Path | Contents |
|---|---|
| `src/prepare.py` | Thread reconstruction, reply merging, cleaning, temporal split |
| `src/taxonomy.py`, `src/silver.py` | Cluster report, curated 10-intent taxonomy, silver labels |
| `src/agent.py`, `src/guard.py`, `src/retrieval.py` | System A |
| `src/baselines.py`, `src/tfidf_intent.py` | B0 and B1 |
| `src/run_systems.py`, `src/evaluate.py`, `src/metrics.py`, `src/reply_checks.py` | Evaluation harness |
| `src/judge.py`, `src/run_judge.py`, `src/judge_agreement.py` | LLM judge and its validation |
| `src/llm.py`, `src/providers.py`, `config/models.json` | Role-based model access, cache, modes |
| `tools/label_server.py`, `tools/score_server.py` | Labelling and blind scoring UIs |
| `data/golden/` | The 200 golden cases, practice round, sampling files |
| `cache/` | Every model output, keyed by model, input and parameters |
| `artifacts/` | Predictions, results, taxonomy and silver labels |

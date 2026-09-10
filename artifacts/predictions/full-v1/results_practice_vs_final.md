# Evaluation results: practice round vs final labels

Labelled cases per subset: {'headline_uniform': 130, 'targeted': 70, 'sensitivity_uniform_unexposed': 73} (labels read for scoring only)
Run: git `b42ef5d`, mode `offline`, models {'embedding': 'sentence-transformers:all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'generator': 'ollama:qwen2.5:3b'}

## Secondary: practice round vs final labels (57 practice-exposed cases)

- Final labels saved from a practice pre-set: 57; changed at least one pre-set field: 8
- Escalation decision, practice vs final: agreement 0.9623, kappa 0.9246 (n=53)

| Pre-set field | Pre-set on | Changed on review |
|---|---|---|
| ba_reply_ok | 53 | 3 |
| escalate | 53 | 2 |
| intent | 45 | 3 |
| note | 15 | 1 |
| reason | 25 | 1 |


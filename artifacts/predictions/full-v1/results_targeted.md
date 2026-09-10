# Evaluation results: Targeted 70 (reported separately, not part of the headline)

Labelled cases per subset: {'headline_uniform': 130, 'targeted': 70, 'sensitivity_uniform_unexposed': 73} (labels read for scoring only)
Run: git `b42ef5d`, mode `offline`, models {'embedding': 'sentence-transformers:all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'generator': 'ollama:qwen2.5:3b'}

## Targeted 70 (reported separately, not part of the headline)

| System | n | Intent acc | Macro-F1 | Must-escalate recall | Unsafe auto-send | Coverage | Auto-sent replies passing checks | Valid outputs |
|---|---|---|---|---|---|---|---|---|
| A+gate | 70 | 0.44 [0.33, 0.56] | 0.30 [0.21, 0.40] | 0.87 [0.76, 0.95] | 0.27 [0.10, 0.47] | 0.31 [0.21, 0.43] | 1.00 [1.00, 1.00] | 0.99 |
| A-no-guard | 70 | 0.44 [0.33, 0.56] | 0.30 [0.21, 0.40] | 0.38 [0.24, 0.53] | 0.56 [0.42, 0.69] | 0.71 [0.60, 0.81] | 0.80 [0.69, 0.91] | 0.99 |
| A-no-retrieval | 70 | 0.50 [0.39, 0.61] | 0.42 [0.29, 0.54] | 0.73 [0.60, 0.86] | 0.36 [0.20, 0.52] | 0.47 [0.36, 0.59] | 0.64 [0.47, 0.80] | 1.00 |
| A | 70 | 0.44 [0.33, 0.56] | 0.30 [0.21, 0.40] | 0.84 [0.72, 0.94] | 0.27 [0.11, 0.44] | 0.37 [0.27, 0.49] | 0.85 [0.71, 0.97] | 0.99 |
| B0-auto-all | 70 | 0.09 [0.03, 0.16] | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.00] | 0.64 [0.53, 0.76] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 |
| B0-escalate-all | 70 | 0.09 [0.03, 0.16] | 0.02 [0.01, 0.03] | 1.00 [1.00, 1.00] | n/a | 0.00 [0.00, 0.00] | n/a | 1.00 |
| B1 | 70 | 0.39 [0.27, 0.50] | 0.35 [0.24, 0.46] | 0.69 [0.55, 0.82] | 0.40 [0.24, 0.55] | 0.50 [0.39, 0.61] | 0.91 [0.82, 1.00] | 1.00 |

*A+gate uses the reply checks as its gate, so its auto-sent replies pass them by construction. Judge or human scores are the independent measure of its replies.*


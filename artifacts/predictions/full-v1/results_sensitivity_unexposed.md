# Evaluation results: Sensitivity: uniform cases not seen in the practice round

Labelled cases per subset: {'headline_uniform': 130, 'targeted': 70, 'sensitivity_uniform_unexposed': 73} (labels read for scoring only)
Run: git `b42ef5d`, mode `offline`, models {'embedding': 'sentence-transformers:all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'generator': 'ollama:qwen2.5:3b'}

## Sensitivity: uniform cases not seen in the practice round

| System | n | Intent acc | Macro-F1 | Must-escalate recall | Unsafe auto-send | Coverage | Auto-sent replies passing checks | Valid outputs |
|---|---|---|---|---|---|---|---|---|
| A+gate | 73 | 0.44 [0.33, 0.56] | 0.47 [0.30, 0.56] | 0.61 [0.45, 0.76] | 0.33 [0.19, 0.47] | 0.59 [0.48, 0.70] | 1.00 [1.00, 1.00] | 0.99 |
| A-no-guard | 73 | 0.44 [0.33, 0.56] | 0.47 [0.30, 0.56] | 0.25 [0.12, 0.40] | 0.43 [0.30, 0.55] | 0.86 [0.78, 0.93] | 0.83 [0.72, 0.91] | 0.99 |
| A-no-retrieval | 73 | 0.48 [0.37, 0.59] | 0.52 [0.36, 0.63] | 0.33 [0.18, 0.49] | 0.41 [0.29, 0.55] | 0.79 [0.70, 0.89] | 0.76 [0.65, 0.86] | 1.00 |
| A | 73 | 0.44 [0.33, 0.56] | 0.47 [0.30, 0.56] | 0.47 [0.32, 0.63] | 0.37 [0.24, 0.50] | 0.71 [0.60, 0.81] | 0.83 [0.72, 0.92] | 0.99 |
| B0-auto-all | 73 | 0.10 [0.03, 0.16] | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.00] | 0.49 [0.37, 0.60] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 |
| B0-escalate-all | 73 | 0.10 [0.03, 0.16] | 0.02 [0.01, 0.03] | 1.00 [1.00, 1.00] | n/a | 0.00 [0.00, 0.00] | n/a | 1.00 |
| B1 | 73 | 0.58 [0.48, 0.68] | 0.50 [0.39, 0.65] | 0.28 [0.14, 0.43] | 0.43 [0.31, 0.56] | 0.82 [0.73, 0.90] | 0.87 [0.78, 0.95] | 1.00 |

*A+gate uses the reply checks as its gate, so its auto-sent replies pass them by construction. Judge or human scores are the independent measure of its replies.*


# Evaluation results: Headline: 130 uniform cases

Labelled cases per subset: {'headline_uniform': 130, 'targeted': 70, 'sensitivity_uniform_unexposed': 73} (labels read for scoring only)
Run: git `b42ef5d`, mode `offline`, models {'embedding': 'sentence-transformers:all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'generator': 'ollama:qwen2.5:3b'}

## Headline: 130 uniform cases

| System | n | Intent acc | Macro-F1 | Must-escalate recall | Unsafe auto-send | Coverage | Auto-sent replies passing checks | Valid outputs |
|---|---|---|---|---|---|---|---|---|
| A+gate | 130 | 0.40 [0.32, 0.48] | 0.42 [0.30, 0.50] | 0.58 [0.47, 0.70] | 0.36 [0.25, 0.46] | 0.58 [0.50, 0.67] | 1.00 [1.00, 1.00] | 0.98 |
| A-no-guard | 130 | 0.40 [0.32, 0.48] | 0.42 [0.30, 0.50] | 0.22 [0.12, 0.32] | 0.45 [0.35, 0.54] | 0.88 [0.82, 0.93] | 0.78 [0.70, 0.85] | 0.98 |
| A-no-retrieval | 130 | 0.47 [0.38, 0.55] | 0.52 [0.40, 0.60] | 0.31 [0.20, 0.43] | 0.43 [0.34, 0.52] | 0.81 [0.74, 0.87] | 0.76 [0.68, 0.84] | 1.00 |
| A | 130 | 0.40 [0.32, 0.48] | 0.42 [0.30, 0.50] | 0.42 [0.29, 0.54] | 0.39 [0.29, 0.49] | 0.75 [0.67, 0.82] | 0.78 [0.70, 0.86] | 0.98 |
| B0-auto-all | 130 | 0.09 [0.05, 0.15] | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.00] | 0.50 [0.42, 0.58] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 |
| B0-escalate-all | 130 | 0.09 [0.05, 0.15] | 0.02 [0.01, 0.03] | 1.00 [1.00, 1.00] | n/a | 0.00 [0.00, 0.00] | n/a | 1.00 |
| B1 | 130 | 0.54 [0.45, 0.62] | 0.48 [0.40, 0.56] | 0.26 [0.16, 0.38] | 0.44 [0.35, 0.54] | 0.83 [0.76, 0.88] | 0.87 [0.80, 0.93] | 1.00 |

*A+gate uses the reply checks as its gate, so its auto-sent replies pass them by construction. Judge or human scores are the independent measure of its replies.*

## Best operating point under the 5% unsafe-auto bar (headline)

| System | Risk signal | Threshold | Coverage | Unsafe auto-send |
|---|---|---|---|---|
| A+gate | tfidf_support | 0.7867 | 0.0385 | 0.0 |
| A+gate | retrieval_top_score | 0.7921 | 0.0462 | 0.0 |
| A+gate | intent_confidence | all_escalated | 0.0 | None |
| A-no-guard | tfidf_support | 0.9591 | 0.0154 | 0.0 |
| A-no-guard | retrieval_top_score | 0.9327 | 0.0077 | 0.0 |
| A-no-guard | intent_confidence | all_escalated | 0.0 | None |
| A-no-retrieval | tfidf_support | 0.9084 | 0.0231 | 0.0 |
| A-no-retrieval | intent_confidence | all_escalated | 0.0 | None |
| A | tfidf_support | 0.9084 | 0.0231 | 0.0 |
| A | retrieval_top_score | 0.7855 | 0.0615 | 0.0 |
| A | intent_confidence | all_escalated | 0.0 | None |
| B0-auto-all | intent_confidence | all_escalated | 0.0 | None |
| B0-escalate-all | intent_confidence | all_escalated | 0.0 | None |
| B1 | intent_confidence | 0.9084 | 0.0231 | 0.0 |

## Paired comparisons (headline; difference = first minus second)

- **A vs B1** (n=130): intent_accuracy -0.14 [-0.25, -0.03]; unsafe_auto_rate -0.05 [-0.09, -0.02]; coverage -0.08 [-0.14, -0.04]
- **A vs B0-auto-all** (n=130): intent_accuracy +0.31 [+0.22, +0.38]; unsafe_auto_rate -0.11 [-0.17, -0.06]; coverage -0.25 [-0.33, -0.18]
- **A+gate vs A** (n=130): intent_accuracy +0.00 [+0.00, +0.00]; unsafe_auto_rate -0.04 [-0.09, +0.01]; coverage -0.16 [-0.23, -0.10]
- **A vs A-no-retrieval** (n=130): intent_accuracy -0.07 [-0.15, +0.01]; unsafe_auto_rate -0.04 [-0.08, -0.00]; coverage -0.06 [-0.12, -0.02]
- **A vs A-no-guard** (n=130): intent_accuracy +0.00 [+0.00, +0.00]; unsafe_auto_rate -0.06 [-0.10, -0.01]; coverage -0.13 [-0.19, -0.08]


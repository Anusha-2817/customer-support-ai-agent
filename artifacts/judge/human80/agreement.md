# Judge vs human agreement

Judge: `ollama:llama3.2:3b` | human-scored items: 80

## Reported: 50 test items (n=49)

| Dimension | Quadratic kappa | Exact agreement | Judge minus human |
|---|---|---|---|
| relevance | 0.0505 | 0.1224 | -0.6327 |
| groundedness | -0.0366 | 0.4694 | 0.5918 |
| actionability | 0.0556 | 0.3673 | -0.3673 |
| tone | -0.0283 | 0.1837 | -0.7551 |
| public_safety | -0.0576 | 0.8163 | -0.1224 |

Sendable: kappa 0.0354, accuracy 0.5918, human rate 0.4082, judge rate 0.0816; confusion (human rows yes/no, judge columns yes/no) [[2, 18], [2, 27]]
Bias by source (judge minus human, mean over dimensions): {'A': {'n': 32, 'judge_minus_human': -0.375}, 'B0-auto-all': {'n': 1, 'judge_minus_human': -0.4}, 'B1': {'n': 16, 'judge_minus_human': -0.0125}}
Verbosity probe (correlation of reply length with the judge-minus-human gap): 0.0294

## Dev items (may be used to tune the rubric) (n=30)

| Dimension | Quadratic kappa | Exact agreement | Judge minus human |
|---|---|---|---|
| relevance | 0.1762 | 0.2 | -0.4 |
| groundedness | 0.0 | 0.5667 | 0.4333 |
| actionability | 0.313 | 0.5 | -0.1333 |
| tone | -0.0515 | 0.2667 | -0.6 |
| public_safety | -0.0825 | 0.7333 | -0.2 |

Sendable: kappa 0.0667, accuracy 0.5333, human rate 0.5, judge rate 0.0333; confusion (human rows yes/no, judge columns yes/no) [[1, 14], [0, 15]]
Bias by source (judge minus human, mean over dimensions): {'A': {'n': 12, 'judge_minus_human': -0.3833}, 'B0-auto-all': {'n': 4, 'judge_minus_human': 0.0}, 'B1': {'n': 14, 'judge_minus_human': -0.0571}}
Verbosity probe (correlation of reply length with the judge-minus-human gap): -0.0683

## Self-preference probe (test items)

{'independent_judge': {'A': -0.375, 'B1': -0.0125}, 'self_judge': {'A': -0.8909, 'B1': -0.7875}, 'self_preference': 0.2591, 'reading': 'positive = the reply-writing model favours its own replies more than the independent judge does, relative to the human'}


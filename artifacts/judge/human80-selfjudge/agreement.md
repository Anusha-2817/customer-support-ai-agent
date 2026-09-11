# Judge vs human agreement

Judge: `ollama:qwen2.5:3b` | human-scored items: 80

## Reported: 50 test items (n=50)

| Dimension | Quadratic kappa | Exact agreement | Judge minus human |
|---|---|---|---|
| relevance | 0.0634 | 0.18 | -1.24 |
| groundedness | -0.1858 | 0.16 | -0.84 |
| actionability | 0.1492 | 0.24 | -1.04 |
| tone | 0.0761 | 0.24 | -0.92 |
| public_safety | -0.055 | 0.72 | -0.26 |

Sendable: kappa 0.0548, accuracy 0.6, human rate 0.42, judge rate 0.02; confusion (human rows yes/no, judge columns yes/no) [[1, 20], [0, 29]]
Bias by source (judge minus human, mean over dimensions): {'A': {'n': 33, 'judge_minus_human': -0.8909}, 'B0-auto-all': {'n': 1, 'judge_minus_human': -1.0}, 'B1': {'n': 16, 'judge_minus_human': -0.7875}}
Verbosity probe (correlation of reply length with the judge-minus-human gap): 0.3722

## Dev items (may be used to tune the rubric) (n=30)

| Dimension | Quadratic kappa | Exact agreement | Judge minus human |
|---|---|---|---|
| relevance | 0.0669 | 0.2667 | -1.2333 |
| groundedness | -0.0884 | 0.1667 | -0.9333 |
| actionability | 0.1189 | 0.3 | -0.8667 |
| tone | 0.0643 | 0.4 | -0.3667 |
| public_safety | -0.0714 | 0.8 | -0.1333 |

Sendable: kappa 0.0, accuracy 0.5, human rate 0.5, judge rate 0.0; confusion (human rows yes/no, judge columns yes/no) [[0, 15], [0, 15]]
Bias by source (judge minus human, mean over dimensions): {'A': {'n': 12, 'judge_minus_human': -0.8833}, 'B0-auto-all': {'n': 4, 'judge_minus_human': -0.95}, 'B1': {'n': 14, 'judge_minus_human': -0.4857}}
Verbosity probe (correlation of reply length with the judge-minus-human gap): 0.0383


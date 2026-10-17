# Learning to Rank on MSLR-WEB10k: Pointwise vs. Pairwise

A small, end-to-end learning-to-rank (LTR) exercise in a single Jupyter notebook. The goal is not a state-of-the-art ranker, but to show the reasoning behind each step of an LTR workflow: data inspection, leakage-safe splitting, evaluation choices, baselines, and a comparison of pointwise and pairwise training.

**Notebook:** [`learning_to_rank.ipynb`](learning_to_rank.ipynb)

## What's in the notebook

- **Data:** a 100-query sample of [MSLR-WEB10k](https://huggingface.co/datasets/philipphager/MSLR-WEB10k) (web search queries, each with candidate documents, 136 numeric features per document, relevance labels 0 to 4), reshaped to one row per (query, document).
- **Preprocessing:** split by query id (with an assertion that no query appears in both train and test), and min-max scaling fit on train only, without clipping test values.
- **Evaluation, written from scratch:** NDCG@10 (exponential gain, log discount, per-query averaging, undefined queries dropped) and MSE, both with unit tests, plus a test showing where the two metrics disagree.
- **Baselines:** random scores, best single feature (selected on train only), and predict-the-train-mean.
- **Pointwise models:** linear regression, then Ridge after diagnosing overfitting from train vs. test results.
- **Pairwise model:** a linear RankNet in plain numpy (sigmoid of score differences, per-pair SGD), a one-pair sanity check, and an exact solve with logistic regression on pair differences to check whether the optimizer was the limit.

## Results (100 queries, 20 test queries)

| Method | Test NDCG@10 | Test MSE |
|---|---|---|
| Random | 0.200 | - |
| Best single feature (f108) | 0.357 | - |
| Predict-the-train-mean | - | 0.651 |
| Linear regression | 0.405 | 1.043 |
| Ridge (alpha=1.0) | 0.428 | 0.561 |
| Pairwise RankNet (numpy SGD) | 0.425 | 8.994 |
| Pairwise exact (logistic regression) | 0.450 | 1.703 |

## Key findings

- **Baselines first.** Plain linear regression looked fine on NDCG but was worse than predicting the mean on MSE. Without the baseline this would have been easy to miss.
- **Train vs. test diagnosed the problem.** Train MSE was half of test MSE, which is overfitting. Ridge fixed most of it.
- **The metric must match the goal.** Pairwise scores have no 0-4 scale, so their MSE is meaningless, while NDCG is the right metric.
- **The hand-written SGD was not optimal.** Solving the same objective exactly gave a lower pair log-loss (0.569 vs. 0.599), which also shows that a linear RankNet is logistic regression on pair differences.
- **Noise caveat.** With only 19 usable test queries, NDCG differences of 0.02 to 0.03 between the models are within noise, so the notebook does not claim that one approach ranks better.

## How to run

```bash
pip install pandas pyarrow numpy scikit-learn jupyter
jupyter notebook learning_to_rank.ipynb
```

The notebook reads the dataset directly from Hugging Face with `pd.read_parquet(url)`, so it needs an internet connection. Pandas reads the whole parquet file before the notebook keeps the first `N_QUERIES` queries, so the first load can take a while and use a few hundred MB of memory. The dataset is not included in this repository.

## Limitations and next steps

- Small sample and a single train/test split; the next step would be more queries and averaging over several query-level splits or cross-validation.
- Linear models only (no feature interactions); a non-linear scorer or gradient-boosted trees with a ranking objective would be the natural extension.
- Hyperparameters (`alpha`, epochs, learning rate, pairs per query) were not tuned on a validation set.
- The SGD could be improved (learning-rate decay, mini-batches), and LambdaRank would weight each pair by its effect on NDCG.

## A note on AI assistance

The code was drafted with AI assistance (Claude). I specified the requirements, reviewed and tested each part, and wrote the explanations and analysis in the notebook.

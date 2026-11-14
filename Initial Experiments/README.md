# Initial MovieLens experiments

This folder contains a first, reproducible run of ItemKNN and BPR, with an optional LightGCN implementation and API-backed LLM reranker. LightGCN is currently disabled while investigating a PyTorch sparse-tensor warning. The notebook and Python runner use the prepared MovieLens 100K files already in `RecBole_DSAIT4335/dataset/ml-100k`; they do not download data.

## Run

Open `MovieLens_Initial_Experiments.ipynb` in Jupyter and run all cells, or run:

```powershell
python "Initial Experiments\run_initial_experiments.py"
```

The run writes its machine-readable outputs under `outputs/`. The notebook reuses the same runner. For the LLM step, set `LLM_API_KEY` and optionally `LLM_BASE_URL` and `LLM_MODEL` before running. It sends the selected user's genre counts and the top 100 BPR candidates' titles/genres to an OpenAI-compatible `/chat/completions` endpoint. It checks that the returned ordering contains every candidate exactly once. This environment had no API key or local LLM endpoint, so the LLM call was skipped and no LLM top-10 list is claimed.

## Initial setup and interpretation

- MovieLens 100K: 100,000 ratings, 943 users, 1,682 movies. Ratings of 4 or 5 count as positive feedback.
- Evaluation: each user with at least two positive ratings contributes their latest positive interaction as a single held-out target (942 users). For evaluated users, interactions at or after that target timestamp are censored from training; all training-seen movies are removed from recommendation candidates.
- ItemKNN uses cosine similarity between binary user-item positive-feedback columns.
- BPR learns user and item factors from pairwise positive-vs-negative samples.
- LightGCN code learns factors over a normalized user-item graph and averages embeddings across three propagation layers. It is not included in the current run; the sparse graph path emits a PyTorch warning mentioning possible memory errors. The warning appeared during tensor construction, and the script now skips that path.
- The LLM step is a reranker over BPR's top 100 candidates, not an independent all-catalog recommender. It uses titles and genre metadata plus the selected user's training genre counts.
- Hyperparameters are starter values, not tuned. The reported metrics are an initial baseline, not a final model comparison. A one-positive-per-user holdout is simple but does not measure repeated future interactions or global temporal drift across users.

The current CPU run gave BPR HR@10 0.123 / NDCG@10 0.061 and ItemKNN 0.101 / 0.056. In an earlier run before LightGCN was disabled, it scored HR@10 0.079 / NDCG@10 0.041; that result is retained here only as context and is not part of the current output tables. `outputs/top10_overlap.csv` reports overlap for the two models currently run. Differences on this small initial split should guide follow-up experiments rather than support a final conclusion.

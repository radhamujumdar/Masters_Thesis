"""Small, reproducible MovieLens 100K recommendation experiments.

The LLM reranker is an optional OpenAI-compatible chat-completions call. Set
LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL) to run that step.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "RecBole_DSAIT4335" / "dataset" / "ml-100k"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
RATING_THRESHOLD = 4.0
TOP_K = 10
LLM_CANDIDATE_K = 100
DEVICE = torch.device("cpu")
RUN_LIGHTGCN = False  # Disabled while investigating the sparse-tensor warning.


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_split():
    interactions = pd.read_csv(DATA_DIR / "ml-100k.inter", sep="\t")
    items = pd.read_csv(DATA_DIR / "ml-100k.item", sep="\t")
    users = pd.read_csv(DATA_DIR / "ml-100k.user", sep="\t")
    for frame in (interactions, items, users):
        frame.columns = [str(column).split(":", 1)[0] for column in frame.columns]
    interactions["timestamp"] = interactions["timestamp"].astype("int64")
    interactions["rating"] = interactions["rating"].astype(float)
    user_ids = sorted(interactions.user_id.unique())
    item_ids = sorted(items.item_id.unique())
    user_to_idx = {v: i for i, v in enumerate(user_ids)}
    item_to_idx = {v: i for i, v in enumerate(item_ids)}
    interactions["u"] = interactions.user_id.map(user_to_idx)
    interactions["i"] = interactions.item_id.map(item_to_idx)
    items = items.drop_duplicates("item_id").set_index("item_id").reindex(item_ids).reset_index()
    positive = interactions[interactions.rating >= RATING_THRESHOLD]

    # One latest positive per user with at least two positives is held out.
    heldout_indices = []
    for _, group in positive.groupby("user_id", sort=False):
        if len(group) >= 2:
            heldout_indices.append(group.sort_values(["timestamp", "item_id"]).index[-1])
    heldout = interactions.loc[heldout_indices].copy()
    # For evaluated users, censor all events at or after the target timestamp so
    # the user's own future ratings cannot leak into their training history.
    train_mask = pd.Series(True, index=interactions.index)
    for row in heldout.itertuples():
        later_or_tied = (interactions.u == int(row.u)) & (interactions.timestamp >= int(row.timestamp))
        train_mask.loc[later_or_tied[later_or_tied].index] = False
    train = interactions.loc[train_mask].copy()
    train_pos = train[train.rating >= RATING_THRESHOLD].copy()
    n_users, n_items = len(user_ids), len(item_ids)
    matrix = sp.csr_matrix(
        (np.ones(len(train_pos), dtype=np.float32), (train_pos.u, train_pos.i)),
        shape=(n_users, n_items),
    )
    seen = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (train.u, train.i)),
        shape=(n_users, n_items),
    )
    test = {
        int(row.u): int(row.i)
        for row in heldout.itertuples()
        if float(row.rating) >= RATING_THRESHOLD
    }
    return interactions, train, train_pos, heldout, items, users, user_ids, item_ids, matrix, seen, test


def fit_itemknn(matrix: sp.csr_matrix):
    # Cosine similarity on binary positive-feedback vectors.
    co = (matrix.T @ matrix).toarray().astype(np.float32)
    norms = np.sqrt(np.diag(co))
    denom = np.outer(norms, norms)
    sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
    np.fill_diagonal(sim, 0.0)
    return sim


def sample_negatives(users: np.ndarray, positive_mask: np.ndarray, n_items: int):
    negatives = np.random.randint(0, n_items, size=len(users))
    bad = positive_mask[users, negatives]
    while bad.any():
        negatives[bad] = np.random.randint(0, n_items, size=int(bad.sum()))
        bad = positive_mask[users, negatives]
    return negatives


class MF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)

    def forward(self, users, pos, neg):
        u, p, n = self.user(users), self.item(pos), self.item(neg)
        pos_score = (u * p).sum(-1)
        neg_score = (u * n).sum(-1)
        reg = (u.square().sum(-1) + p.square().sum(-1) + n.square().sum(-1)).mean()
        return F.softplus(neg_score - pos_score).mean() + 1e-4 * reg


def fit_bpr(train_pos: pd.DataFrame, n_users: int, n_items: int, epochs: int = 20):
    model = MF(n_users, n_items).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    users = train_pos.u.to_numpy(dtype=np.int64)
    items = train_pos.i.to_numpy(dtype=np.int64)
    pos_mask = np.zeros((n_users, n_items), dtype=bool)
    pos_mask[users, items] = True
    for _ in range(epochs):
        order = np.random.permutation(len(users))
        for start in range(0, len(order), 2048):
            batch = order[start : start + 2048]
            bu = users[batch]
            bp = items[batch]
            bn = sample_negatives(bu, pos_mask, n_items)
            opt.zero_grad(set_to_none=True)
            loss = model(torch.as_tensor(bu), torch.as_tensor(bp), torch.as_tensor(bn))
            loss.backward()
            opt.step()
    with torch.no_grad():
        scores = model.user.weight @ model.item.weight.T
    return scores.cpu().numpy()


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.layers = layers
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)

    def propagate(self, graph):
        initial = torch.cat([self.user.weight, self.item.weight], dim=0)
        states = [initial]
        x = initial
        for _ in range(self.layers):
            x = torch.sparse.mm(graph, x)
            states.append(x)
        final = torch.stack(states, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]


def normalized_bipartite_graph(matrix: sp.csr_matrix):
    n_users, n_items = matrix.shape
    upper = sp.hstack([sp.csr_matrix((n_users, n_users)), matrix], format="csr")
    lower = sp.hstack([matrix.T, sp.csr_matrix((n_items, n_items))], format="csr")
    adj = sp.vstack([upper, lower], format="coo")
    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    inv_sqrt[degree > 0] = degree[degree > 0] ** -0.5
    values = adj.data * inv_sqrt[adj.row] * inv_sqrt[adj.col]
    indices = torch.as_tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
    vals = torch.as_tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, vals, adj.shape).coalesce().to(DEVICE)


def fit_lightgcn(train_pos: pd.DataFrame, matrix: sp.csr_matrix, n_users: int, n_items: int, epochs: int = 25):
    model = LightGCN(n_users, n_items).to(DEVICE)
    graph = normalized_bipartite_graph(matrix)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    u_np = train_pos.u.to_numpy(dtype=np.int64)
    i_np = train_pos.i.to_numpy(dtype=np.int64)
    pos_mask = np.zeros((n_users, n_items), dtype=bool)
    pos_mask[u_np, i_np] = True
    u = torch.as_tensor(u_np)
    i = torch.as_tensor(i_np)
    for _ in range(epochs):
        neg = torch.as_tensor(sample_negatives(u_np, pos_mask, n_items))
        opt.zero_grad(set_to_none=True)
        user_e, item_e = model.propagate(graph)
        pos_score = (user_e[u] * item_e[i]).sum(-1)
        neg_score = (user_e[u] * item_e[neg]).sum(-1)
        reg = (model.user(u).square().sum(-1) + model.item(i).square().sum(-1) + model.item(neg).square().sum(-1)).mean()
        loss = F.softplus(neg_score - pos_score).mean() + 1e-4 * reg
        loss.backward()
        opt.step()
    with torch.no_grad():
        user_e, item_e = model.propagate(graph)
        scores = user_e @ item_e.T
    return scores.cpu().numpy()


def score_itemknn(sim: np.ndarray, matrix: sp.csr_matrix):
    return (matrix @ sim).astype(np.float32)


def evaluate(scores, seen: sp.csr_matrix, test: dict, k: int = TOP_K):
    hits, ndcgs, ranks = [], [], []
    for u, target in test.items():
        row = scores[u].copy()
        seen_items = seen.indices[seen.indptr[u] : seen.indptr[u + 1]]
        row[seen_items] = -np.inf
        target_score = row[target]
        rank = 1 + int(np.sum(row > target_score))
        ranks.append(rank)
        hit = rank <= k
        hits.append(float(hit))
        ndcgs.append(1.0 / math.log2(rank + 1) if hit else 0.0)
    return {"HR@10": float(np.mean(hits)), "NDCG@10": float(np.mean(ndcgs)), "evaluated_users": len(test), "median_target_rank": float(np.median(ranks))}


def topk_for_user(model_scores, user_idx, seen, item_ids, items, k=TOP_K):
    row = np.asarray(model_scores[user_idx]).astype(float).copy()
    already = seen.indices[seen.indptr[user_idx] : seen.indptr[user_idx + 1]]
    row[already] = -np.inf
    take = np.argsort(-row)[:k]
    details = items.set_index("item_id")
    rows = []
    for rank, idx in enumerate(take, 1):
        item_id = item_ids[int(idx)]
        item = details.loc[item_id]
        rows.append({"rank": rank, "item_id": item_id, "movie_title": item.movie_title, "genres": item["class"], "score": float(row[idx])})
    return rows


def genres_for_profile(train_pos, user_idx, item_ids, items):
    liked = train_pos.loc[train_pos.u == user_idx, "i"].map(lambda i: item_ids[int(i)])
    item_data = items.set_index("item_id")
    counts = {}
    for item_id in liked:
        for genre in str(item_data.loc[item_id, "class"]).split():
            counts[genre] = counts.get(genre, 0) + 1
    return sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:8]


def call_llm_reranker(profile, candidates):
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "Skipped: set LLM_API_KEY to enable the API-backed reranker."
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    prompt_candidates = [
        {"candidate_id": idx, "title": x["movie_title"], "genres": x["genres"]}
        for idx, x in enumerate(candidates)
    ]
    prompt = (
        "Rerank these movie candidates for a recommender-system user. Use only the supplied "
        "history genre counts and candidate metadata. Prefer likely user relevance while keeping "
        "some genre variety. Do not invent movies. Return JSON only as {\"ranking\":[candidate_id,...]} "
        "with every candidate_id exactly once.\n"
        f"History genre counts: {json.dumps(profile)}\n"
        f"Candidates: {json.dumps(prompt_candidates, ensure_ascii=False)}"
    )
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            response_json = json.loads(response.read().decode())
        content = response_json["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return None, "LLM returned no JSON ranking."
        order = json.loads(match.group(0))["ranking"]
        if sorted(order) != list(range(len(candidates))):
            return None, "LLM ranking failed candidate-ID validation."
        return [candidates[int(i)] for i in order], f"Completed with {model} via {base_url}."
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError) as exc:
        return None, f"LLM call failed: {type(exc).__name__}: {exc}"


def run_experiment():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    (interactions, train, train_pos, heldout, items, users, user_ids, item_ids,
     matrix, seen, test) = load_and_split()
    n_users, n_items = matrix.shape
    scores = {}
    scores["ItemKNN"] = score_itemknn(fit_itemknn(matrix), matrix)
    scores["BPR"] = fit_bpr(train_pos, n_users, n_items)
    if RUN_LIGHTGCN:
        scores["LightGCN"] = fit_lightgcn(train_pos, matrix, n_users, n_items)

    # Select the most active user with held-out evaluation data for the example lists.
    activity = train_pos.groupby("u").size()
    example_user = max(test, key=lambda u: (int(activity.get(u, 0)), -u))
    example_external_id = user_ids[example_user]
    metrics = {name: evaluate(values, seen, test) for name, values in scores.items()}
    pd.DataFrame([{"algorithm": name, **m} for name, m in metrics.items()]).to_csv(OUT_DIR / "metrics.csv", index=False)

    top_lists = {}
    for name, values in scores.items():
        top_lists[name] = topk_for_user(values, example_user, seen, item_ids, items)
        for row in top_lists[name]:
            row["algorithm"] = name
            row["user_id"] = example_external_id
    pd.DataFrame([row for rows in top_lists.values() for row in rows]).to_csv(OUT_DIR / "top10_recommendations.csv", index=False)

    candidate_model = "LightGCN" if "LightGCN" in scores else "BPR"
    candidates = topk_for_user(scores[candidate_model], example_user, seen, item_ids, items, k=LLM_CANDIDATE_K)
    profile = genres_for_profile(train_pos, example_user, item_ids, items)
    llm_result, llm_status = call_llm_reranker(profile, candidates)
    if llm_result:
        llm_top10 = []
        for rank, entry in enumerate(llm_result[:TOP_K], 1):
            llm_top10.append({**entry, "rank": rank, "algorithm": "LLM rerank (LightGCN top-100)", "user_id": example_external_id})
        pd.DataFrame(llm_top10).to_csv(OUT_DIR / "llm_reranked_top10.csv", index=False)
        top_lists["LLM rerank (LightGCN top-100)"] = llm_top10
    pd.DataFrame(candidates).to_csv(OUT_DIR / "llm_candidate_pool.csv", index=False)

    overlaps = []
    names = list(top_lists)
    for i, left in enumerate(names):
        left_set = {r["item_id"] for r in top_lists[left]}
        for right in names[i + 1 :]:
            right_set = {r["item_id"] for r in top_lists[right]}
            overlaps.append({"algorithm_a": left, "algorithm_b": right, "shared_items": len(left_set & right_set), "jaccard_at_10": len(left_set & right_set) / len(left_set | right_set)})
    pd.DataFrame(overlaps).to_csv(OUT_DIR / "top10_overlap.csv", index=False)
    result = {
        "dataset": "MovieLens 100K prepared RecBole TSV",
        "interactions": int(len(interactions)), "users": int(n_users), "items": int(n_items),
        "positive_threshold": RATING_THRESHOLD, "training_positive_interactions": int(len(train_pos)),
        "heldout_users": int(len(test)), "split": "per-user latest positive held out; each evaluated user's events at or after that timestamp are censored from training",
        "example_user_id": int(example_external_id), "example_user_training_positive_count": int(activity.get(example_user, 0)),
        "example_user_top_genres": profile, "metrics": metrics, "models_run": list(scores),
        "llm_candidate_generator": candidate_model, "llm_status": llm_status,
        "runtime_seconds": round(time.perf_counter() - started, 2), "device": str(DEVICE),
        "notes": ["Initial baseline run; hyperparameters are not tuned.", f"LLM reranker uses {candidate_model}'s top-100 candidates and item title/genre metadata.", "LightGCN was skipped while investigating the PyTorch sparse-tensor warning.", "Top-10 lists omit every item in this user's training interaction history."],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result, top_lists


if __name__ == "__main__":
    run_experiment()

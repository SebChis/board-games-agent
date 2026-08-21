"""Semantic search over the sampled player-review corpus."""

import functools

import numpy as np
import pandas as pd

from ._shared import embed, repo_root


@functools.lru_cache(maxsize=1)
def _reviews() -> pd.DataFrame:
    return pd.read_json(
        repo_root() / "data" / "processed" / "reviews_corpus.jsonl", lines=True
    )


@functools.lru_cache(maxsize=1)
def _review_vectors() -> np.ndarray:
    cache_file = repo_root() / "cache" / "review_embeddings.npy"
    if cache_file.exists():
        return np.load(cache_file)
    print("Embedding review corpus (one-time, a few minutes)…")
    vectors = embed(_reviews()["textfield_comment_value"].tolist(), task_type="RETRIEVAL_DOCUMENT")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, vectors)
    return vectors


def search_reviews(query: str, game: str = "", top_k: int = 5) -> dict:
    """Searches player reviews by MEANING, not keywords.

    Use whenever the user asks what players think, feel, or complain about
    for a game. This is opinion, not fact — for rules questions, use
    search_rules / search_errata instead.

    Args:
        query: What to look for, phrased naturally, e.g. "confusing scoring".
        game: Optional exact-ish game title to filter to (case-insensitive,
            partial match). Leave empty to search across all 10 games.
        top_k: How many reviews to return (default 5).

    Returns:
        dict: status, and a list of hits with Title, rating, score, review text.
    """
    df = _reviews()
    vectors = _review_vectors()

    if game:
        mask = df["Title"].str.lower().str.contains(game.lower(), na=False)
        if not mask.any():
            return {"status": "error", "message": f"No reviews found for game {game!r}."}
        idx = df[mask].index.to_numpy()
    else:
        idx = df.index.to_numpy()

    q = embed(query, task_type="RETRIEVAL_QUERY")[0]
    scores = vectors[idx] @ q
    top_local = np.argsort(scores)[::-1][:top_k]
    top_global = idx[top_local]

    hits = []
    for gi, li in zip(top_global, top_local):
        row = df.iloc[int(gi)]
        hits.append({
            "title": row["Title"],
            "rating": row["rating"],
            "score": round(float(scores[li]), 3),
            "review_text": row["textfield_comment_value"],
        })
    return {"status": "success", "hits": hits}
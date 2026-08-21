"""Shared plumbing: repo root, Gemini client, embeddings."""

import functools
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash-lite"


def repo_root() -> Path:
    """Walk upwards until we find the repo root (the folder containing data/processed)."""
    p = Path(__file__).resolve().parent
    while not (p / "data" / "processed" / "games_metadata.csv").exists():
        if p == p.parent:
            raise RuntimeError("Could not locate repo root (data/processed/games_metadata.csv).")
        p = p.parent
    return p


@functools.lru_cache(maxsize=1)
def client() -> genai.Client:
    return genai.Client()


def embed(texts: list[str] | str, task_type: str) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    for attempt in range(7):
        try:
            result = client().models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
            )
            vectors = np.array([e.values for e in result.embeddings])
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        except genai.errors.APIError:
            time.sleep(2**attempt)
    raise RuntimeError("embedding failed after 7 attempts")
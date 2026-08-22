"""Shared plumbing: repo root, Gemini client, embeddings."""

import functools
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

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

load_dotenv(repo_root() / ".env", override=True)

@functools.lru_cache(maxsize=1)
def client() -> genai.Client:
    return genai.Client()


def embed(texts: list[str] | str, task_type: str, batch_size: int = 100) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        result = client().models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )
        vectors = np.array([e.values for e in result.embeddings])
        all_vectors.append(vectors)
        print(f"  embedded {start + len(batch)}/{len(texts)}")

        if start + batch_size < len(texts):
            print("  waiting 62s (free-tier quota is per-minute)...")
            time.sleep(62)

    vectors = np.vstack(all_vectors)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
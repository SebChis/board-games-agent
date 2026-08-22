"""RAG over data/docs/rules and data/docs/errata — chunked per section."""

import functools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .tools._shared import embed, repo_root

INDEX_DIR = None  # set lazily below, after repo_root() is available


def _index_dir() -> Path:
    global INDEX_DIR
    if INDEX_DIR is None:
        INDEX_DIR = repo_root() / "cache"
    return INDEX_DIR


def _vectors_file() -> Path:
    return _index_dir() / "doc_vectors.npy"


def _meta_file() -> Path:
    return _index_dir() / "doc_meta.json"


@functools.lru_cache(maxsize=1)
def _game_slugs() -> dict:
    """Maps a filename slug (e.g. 'ticket_to_ride') to the canonical Title."""
    df = pd.read_csv(repo_root() / "data" / "processed" / "games_metadata.csv")
    slugs = {}
    for title in df["Title"]:
        slug = re.sub(r"[^a-z0-9]", "", title.lower())
        slugs[slug] = title
    return slugs


def _resolve_title(filename_stem: str) -> str:
    """'ticket_to_ride_rules' -> 'Ticket to Ride' (fuzzy match against catalog)."""
    slug = re.sub(r"[^a-z0-9]", "", filename_stem.lower())
    slug = slug.replace("rules", "").replace("errata", "")
    for known_slug, title in _game_slugs().items():
        if known_slug == slug:
            return title
    return filename_stem  # fallback: unresolved, keep as-is


def _chunk_rules(text: str, title: str, source_file: str) -> list[dict]:
    """One chunk per ## section."""
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        header = re.match(r"^##\s*(.+)", part)
        section = header.group(1).strip() if header else "Intro"
        chunks.append({
            "game": title,
            "doc_type": "rules",
            "section": section,
            "source_file": source_file,
            "override": False,
            "text": part,
        })
    return chunks


def _chunk_errata(text: str, title: str, source_file: str) -> list[dict]:
    """One chunk per ### entry (each Q&A / correction stands alone)."""
    parts = re.split(r"\n(?=### )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part or not part.startswith("###"):
            continue
        header = re.match(r"^###\s*(.+)", part)
        section = header.group(1).strip() if header else "Entry"
        chunks.append({
            "game": title,
            "doc_type": "errata",
            "section": section,
            "source_file": source_file,
            "override": "[SUPRASCRIE REGULA DE BAZĂ]" in part,
            "text": part,
        })
    return chunks


def build_index() -> int:
    """Embeds every chunk from data/docs/rules and data/docs/errata."""
    docs_dir = repo_root() / "data" / "docs"
    all_chunks = []

    for path in sorted((docs_dir / "rules").glob("*.md")):
        title = _resolve_title(path.stem)
        all_chunks += _chunk_rules(path.read_text(encoding="utf-8"), title, path.name)

    for path in sorted((docs_dir / "errata").glob("*.md")):
        title = _resolve_title(path.stem)
        all_chunks += _chunk_errata(path.read_text(encoding="utf-8"), title, path.name)

    if not all_chunks:
        raise RuntimeError(f"No chunks produced from {docs_dir}")

    print(f"Embedding {len(all_chunks)} chunks from {docs_dir}…")
    vectors = embed([c["text"] for c in all_chunks], task_type="RETRIEVAL_DOCUMENT")

    _index_dir().mkdir(parents=True, exist_ok=True)
    np.save(_vectors_file(), vectors)
    _meta_file().write_text(json.dumps(all_chunks), encoding="utf-8")
    return len(all_chunks)


@functools.lru_cache(maxsize=1)
def _index() -> tuple[np.ndarray, list[dict]]:
    if not _vectors_file().exists():
        build_index()
    return np.load(_vectors_file()), json.loads(_meta_file().read_text(encoding="utf-8"))


def _search(query: str, doc_type: str, game: str = "", top_k: int = 5) -> dict:
    vectors, meta = _index()

    idx = [
        i for i, m in enumerate(meta)
        if m["doc_type"] == doc_type
        and (not game or game.lower() in m["game"].lower())
    ]
    if not idx:
        return {"status": "error", "message": f"No {doc_type} docs found for game {game!r}."}

    idx = np.array(idx)
    q = embed(query, task_type="RETRIEVAL_QUERY")[0]
    scores = vectors[idx] @ q
    top_local = np.argsort(scores)[::-1][:top_k]
    top_global = idx[top_local]

    hits = []
    for gi, li in zip(top_global, top_local):
        m = meta[int(gi)]
        hits.append({
            "game": m["game"],
            "section": m["section"],
            "source_file": m["source_file"],
            "override": m["override"],
            "score": round(float(scores[li]), 3),
            "text": m["text"],
        })
    return {"status": "success", "hits": hits}


def search_rules(query: str, game: str = "", top_k: int = 5) -> dict:
    """Searches the BASE rulebooks (setup, turn structure, scoring, exceptions).

    Use for "how does X work" / "what's the rule for Y" questions. ALWAYS
    also check search_errata afterwards — a base rule may have been officially
    overridden.

    Args:
        query: What to look for, phrased naturally.
        game: Optional game title filter (partial match). Leave empty to
            search all 10 games — useful when the game isn't known yet.
        top_k: How many chunks to return (default 5).
    """
    return _search(query, doc_type="rules", game=game, top_k=top_k)


def search_errata(query: str, game: str = "", top_k: int = 5) -> dict:
    """Searches official FAQ/errata — corrections and clarifications to the
    base rules, each with a "override" flag (True = this entry replaces a
    base rule; treat it as the authoritative answer over search_rules).

    Args:
        query: What to look for, phrased naturally.
        game: Optional game title filter (partial match).
        top_k: How many entries to return (default 5).
    """
    return _search(query, doc_type="errata", game=game, top_k=top_k)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks → {_vectors_file().relative_to(repo_root())}")
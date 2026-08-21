"""Structured metadata tools — the 10-game catalog."""

import functools

import pandas as pd

from ._shared import repo_root


@functools.lru_cache(maxsize=1)
def _games() -> pd.DataFrame:
    return pd.read_csv(repo_root() / "data" / "processed" / "games_metadata.csv")


def list_games() -> dict:
    """Lists all 10 board games in the collection with basic metadata.

    Use this to see which games exist, or to find the exact title when the
    user gives a partial or misspelled name.

    Returns:
        dict: status, and a list of games with Title, Year, AvgRating, GeekRating.
    """
    games = _games()[["Title", "Year", "AvgRating", "GeekRating"]]
    return {"status": "success", "games": games.to_dict(orient="records")}


def get_game_details(title: str) -> dict:
    """Gets full catalog metadata for one game by title (case-insensitive, partial match ok).

    Args:
        title: The game's title, e.g. "Catan" or "ticket to ride".

    Returns:
        dict: status and the game's metadata (Title, Year, Description, ratings, Link).
    """
    df = _games()
    match = df[df["Title"].str.lower().str.contains(title.lower(), na=False)]
    if match.empty:
        return {
            "status": "error",
            "message": f"No game matching {title!r}. Call list_games to see valid titles.",
        }
    return {"status": "success", **match.iloc[0].to_dict()}
"""
Filters the local BGG Kaggle dataset (boardgames.csv + boardgames_reviews.csv)
down to the 6 games this project uses, and writes clean files to data/processed/.

Only text reviews (non-empty comment) are kept - rating-only rows carry no
semantic content and are useless for the vector search corpus.
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

RAW_GAMES_FILE = RAW_DIR / "boardgames.csv"
RAW_REVIEWS_FILE = RAW_DIR / "boardgames_reviews.csv"

GAMES = [
    "Catan",
    "Ticket to Ride",
    "Pandemic",
    "Carcassonne",
    "Splendor",
    "Azul",
    "7 Wonders",
    "Codenames",
    "King of Tokyo",
    "Wingspan",
]

# Only pull the columns we actually need from the big reviews file -
# skips parsing postdate/rating_tstamp duplicates, cuts memory a lot.
REVIEW_COLUMNS_TO_KEEP = [
    "Game_Id",
    "reviewid",
    "user_pseudouserid",
    "textfield_comment_value",
    "rating",
    "textfield_comment_tstamp",
]


def filter_games_metadata():
    df = pd.read_csv(RAW_GAMES_FILE, low_memory=False)

    mask = df["Title"].str.strip().str.lower().isin([g.lower() for g in GAMES])
    filtered = df[mask].copy()

    print(f"Matched {len(filtered)} / {len(GAMES)} games:")
    print(filtered["Title"].tolist())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "games_metadata.csv"
    filtered.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    return filtered


def filter_reviews(game_ids):
    kept_chunks = []
    total_seen = 0
    total_matched_ids = 0

    for chunk in pd.read_csv(
        RAW_REVIEWS_FILE,
        chunksize=500_000,
        low_memory=False,
        usecols=REVIEW_COLUMNS_TO_KEEP,
    ):
        total_seen += len(chunk)

        # keep only our 6 games
        chunk = chunk[chunk["Game_Id"].isin(game_ids)]
        total_matched_ids += len(chunk)

        # keep only reviews with actual text (drop rating-only rows)
        chunk["textfield_comment_value"] = chunk["textfield_comment_value"].fillna("")
        chunk = chunk[chunk["textfield_comment_value"].str.strip() != ""]

        if len(chunk):
            kept_chunks.append(chunk)

    filtered = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame()

    print(f"Scanned {total_seen:,} rows total")
    print(f"Matched {total_matched_ids:,} rows for our 6 games")
    print(f"Kept {len(filtered):,} rows with non-empty review text")

    out_path = PROCESSED_DIR / "reviews_filtered.jsonl"
    filtered.to_json(out_path, orient="records", lines=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    games_df = filter_games_metadata()
    game_ids = games_df["Game_Id"].tolist()

    filter_reviews(game_ids)
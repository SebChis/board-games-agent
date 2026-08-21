"""
Downsamples the filtered reviews into a manageable, balanced corpus
for the RAG vector store: caps per-game count, filters out very short
low-signal comments, and keeps a mix of long/short + recent/old reviews.
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

IN_PATH = PROCESSED_DIR / "reviews_filtered.jsonl"
OUT_PATH = PROCESSED_DIR / "reviews_corpus.jsonl"

MIN_CHARS = 40          # drop very short, low-signal comments ("nice game")
MAX_PER_GAME = 800       # cap per game, keeps corpus balanced + embeddings cheap
RANDOM_SEED = 42


def main():
    df = pd.read_json(IN_PATH, lines=True)
    print(f"Loaded {len(df):,} reviews")

    # 1. Drop low-signal short comments
    df = df[df["textfield_comment_value"].str.len() >= MIN_CHARS]
    print(f"After min-length filter ({MIN_CHARS} chars): {len(df):,}")

    # 2. Drop exact duplicate text (copy-pasted reviews happen)
    df = df.drop_duplicates(subset="textfield_comment_value")
    print(f"After dedup: {len(df):,}")

    # 3. Per-game distribution before capping
    print("\nPer-game counts before cap:")
    print(df["Game_Id"].value_counts())

    # 4. Stratified sample: cap each game at MAX_PER_GAME, prefer longer
    #    reviews (more likely to contain rules/strategy discussion, not
    #    just "loved it!"), but keep some randomness so it's not only essays.
    df["text_len"] = df["textfield_comment_value"].str.len()

    sampled = []
    for game_id, group in df.groupby("Game_Id"):
        if len(group) <= MAX_PER_GAME:
            sampled.append(group)
            continue
        # take top 40% by length + random sample of the rest, so we mix
        # detailed reviews with shorter/casual ones
        n_top = int(MAX_PER_GAME * 0.4)
        n_random = MAX_PER_GAME - n_top

        top_long = group.nlargest(n_top, "text_len")
        remaining = group.drop(top_long.index)
        random_sample = remaining.sample(n=min(n_random, len(remaining)), random_state=RANDOM_SEED)

        sampled.append(pd.concat([top_long, random_sample]))

    result = pd.concat(sampled, ignore_index=True).drop(columns=["text_len"])
    # după ce ai creat `result`, înainte de result.to_json(...)
    games_df = pd.read_csv(PROCESSED_DIR / "games_metadata.csv")
    result = result.merge(
        games_df[["Game_Id", "Title"]], on="Game_Id", how="left"
    )
    print(f"\nFinal corpus: {len(result):,} reviews")
    print("Per-game counts after cap:")
    print(result["Game_Id"].value_counts())

    result.to_json(OUT_PATH, orient="records", lines=True)
    print(f"\nWrote {OUT_PATH}")

    
if __name__ == "__main__":
    main()
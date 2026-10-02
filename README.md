# Board Rules Agent

>  **Work in progress.** Core data pipeline and RAG retrieval are functional; agent behavior is being tested and refined. Multi-agent critic loop, cross-session memory, and evals suite are not yet implemented.

A stateful AI agent that arbitrates board game rule conflicts, combining structured metadata, semantically-indexed rulebooks/errata, and community reviews — built to demonstrate RAG over heterogeneous data sources, tool orchestration, and session state management.

## What it does

Given a question about any of 10 board games, the agent:
- Looks up catalog facts (ratings, year, description) from structured metadata
- Searches base rulebooks semantically for relevant rules
- Cross-checks official errata/FAQ documents, flagging when an errata entry **overrides** a base rule
- Searches a corpus of player reviews for opinions/sentiment (kept separate from factual rules)
- Never answers from model memory — only from indexed sources, to avoid hallucinated rules

## Games covered

Catan, Ticket to Ride, Pandemic, Carcassonne, Splendor, Azul, 7 Wonders, Codenames, King of Tokyo, Wingspan.

## Architecture

Three heterogeneous data pipelines feed the agent:

| Source | Type | Storage |
|---|---|---|
| Game metadata (ratings, year, description) | Structured | `data/processed/games_metadata.csv` |
| Rulebooks + official errata/FAQ | Unstructured, chunked by section | `data/docs/rules/`, `data/docs/errata/` → vector index |
| Player reviews (sampled, text-only) | Semi-structured | `data/processed/reviews_corpus.jsonl` |

Rulebook and errata documents are chunked per section (`##` for rules, `###` for individual errata entries) and embedded with Gemini's embedding model. Errata chunks carry an `override` flag when they explicitly supersede a base rule, so the agent can prioritize the correct source.

## Project structure
```
.
├── data/
│ ├── docs/
│ │ ├── rules/ # rulebook .md per game
│ │ └── errata/ # FAQ/errata .md per game
│ ├── processed/ # cleaned CSV/JSONL used by the agent
│ └── raw/ # raw Kaggle source files (gitignored)
├── cache/ # generated embeddings + index (gitignored)
├── src/agent/
│ ├── agent.py # orchestrator (LlmAgent, instructions, tool list)
│ ├── retrieval.py # chunking + vector index over rules/errata
│ ├── state.py # session notes ledger
│ └── tools/
│ ├── _shared.py # embedding client, repo_root, batching
│ ├── catalog.py # list_games, get_game_details
│ ├── reviews.py # semantic search over review corpus
│ ├── ingest_bgg.py # filters raw Kaggle dataset to the 10 games
│ └── sample_reviews.py # stratified sampling of reviews corpus
├── evals/ # eval cases (not yet populated)
└── requirements.txt
```


## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add GOOGLE_API_KEY
```

Build the vector index (one-time, ~1 minute due to free-tier rate limits):

```bash
python -m src.agent.retrieval
```

Run the agent:

```bash
cd src
adk web
```

## Data pipeline

Source data comes from a Kaggle BoardGameGeek dump (metadata + reviews), since BGG's live XML API now requires an approved application token. Reviews are filtered to the 10 games, deduplicated, stripped of rating-only (textless) entries, and stratified-sampled (~100 per game) to keep the corpus balanced and embedding-cheap. Rulebook and errata documents were converted from official PDFs/FAQ pages into structured Markdown.

## Status

- [x] Metadata + review ingestion (filtered from Kaggle dataset)
- [x] Rules + errata docs for all 10 games
- [x] Chunked vector index over rules/errata
- [x] Core tools (catalog, review search, rule/errata search)
- [x] Agent orchestration (single agent, tool-calling)
- [x] Session notes ledger (in-session only, not cross-session)
- [ ] Agent behavior testing (in progress)
- [ ] Multi-agent critic loop
- [ ] Cross-session memory
- [ ] Evals suite

"""Session notes ledger: tracks what's been asked/resolved during a session."""

import time

from google.adk.tools import ToolContext


def add_note(game: str, note: str, tool_context: ToolContext) -> dict:
    """Records a resolved rule question in the session's notes ledger.

    Use this after answering a rules question, so later questions in the
    same session can build on prior context instead of re-deriving it.

    Args:
        game: The game the note is about.
        note: A short factual note, e.g. "Errata overrides base rule on farmer scoring."

    Returns:
        dict: status and the updated ledger.
    """
    ledger = list(tool_context.state.get("notes_ledger", []))
    ledger.append({"game": game, "note": note, "ts": time.strftime("%H:%M:%S")})
    tool_context.state["notes_ledger"] = ledger
    return {"status": "success", "ledger": ledger}


def get_notes(tool_context: ToolContext, game: str = "") -> dict:
    """Returns notes recorded so far this session, optionally filtered by game.

    Use at the start of answering a question, to check if it was already
    addressed earlier in this conversation.

    Args:
        game: Optional game title filter.
    """
    ledger = tool_context.state.get("notes_ledger", [])
    if game:
        ledger = [n for n in ledger if game.lower() in n["game"].lower()]
    return {"status": "success", "ledger": ledger}
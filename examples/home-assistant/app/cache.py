"""
app/cache.py
────────────
SQLite-backed cache for tool calls.

Schema
──────
table: tool_cache
  message    TEXT  PRIMARY KEY   -- normalised user message (lower, stripped)
  tool_calls TEXT  NOT NULL      -- JSON list of {name, args}
  hits       INT   DEFAULT 0     -- how many times this entry was replayed
  created_at DATETIME            -- first cached
  last_hit   DATETIME            -- most recent cache hit

Only entries that caused the model to call at least one tool are stored.
Text-only responses are never cached.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path("cache.db")


# ── Lifecycle ──────────────────────────────────────────────────────────────────

def init_db(path: str | Path = _DB_PATH) -> None:
    """Create the database and table if they don't exist."""
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS tool_cache (
                message    TEXT    PRIMARY KEY,
                tool_calls TEXT    NOT NULL,
                hits       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL,
                last_hit   TEXT
            )
        """)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_cached(message: str, path: str | Path = _DB_PATH) -> list[dict] | None:
    """
    Return cached tool calls for *message*, or None on a miss.
    Bumps the hit counter and last_hit timestamp on a cache hit.
    """
    key = _normalise(message)
    with _connect(path) as con:
        row = con.execute(
            "SELECT tool_calls FROM tool_cache WHERE message = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        # Update hit stats
        con.execute(
            "UPDATE tool_cache SET hits = hits + 1, last_hit = ? WHERE message = ?",
            (_now(), key),
        )
    return json.loads(row[0])


def set_cached(
    message: str,
    tool_calls: list[dict],
    path: str | Path = _DB_PATH,
) -> None:
    """
    Persist *tool_calls* (list of {name, args} dicts) keyed by *message*.
    Replaces any previous entry for the same message.
    Only call this when the model actually triggered tools.
    """
    if not tool_calls:
        return                       # guard: never cache a text-only turn
    key = _normalise(message)
    payload = json.dumps(tool_calls)
    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO tool_cache (message, tool_calls, hits, created_at, last_hit)
            VALUES (?, ?, 0, ?, NULL)
            ON CONFLICT(message) DO UPDATE SET
                tool_calls = excluded.tool_calls,
                created_at = excluded.created_at,
                hits       = 0,
                last_hit   = NULL
            """,
            (key, payload, _now()),
        )


def delete_cached(message: str, path: str | Path = _DB_PATH) -> bool:
    """Remove a single cache entry.  Returns True if a row was deleted."""
    key = _normalise(message)
    with _connect(path) as con:
        cur = con.execute(
            "DELETE FROM tool_cache WHERE message = ?", (key,)
        )
        return cur.rowcount > 0


def clear_all(path: str | Path = _DB_PATH) -> int:
    """Wipe the entire cache.  Returns the number of rows deleted."""
    with _connect(path) as con:
        cur = con.execute("DELETE FROM tool_cache")
        return cur.rowcount


def list_entries(path: str | Path = _DB_PATH) -> list[dict]:
    """Return all cache entries (for debugging / admin endpoints)."""
    with _connect(path) as con:
        rows = con.execute(
            "SELECT message, tool_calls, hits, created_at, last_hit FROM tool_cache"
        ).fetchall()
    return [
        {
            "message":    r[0],
            "tool_calls": json.loads(r[1]),
            "hits":       r[2],
            "created_at": r[3],
            "last_hit":   r[4],
        }
        for r in rows
    ]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise(message: str) -> str:
    """Canonical cache key: lowercase, collapsed whitespace."""
    return " ".join(message.lower().split())


def _connect(path: str | Path = _DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

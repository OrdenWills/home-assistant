"""
app/cache.py
────────────
SQLite-backed cache for tool calls.

Schema
──────
table: tool_cache
  message         TEXT  PRIMARY KEY  -- normalised user message (lower, stripped)
  tool_calls      TEXT  NOT NULL     -- JSON list of {name, args}
  device_snapshot TEXT  NOT NULL     -- JSON snapshot of rooms/doors at write time
  hits            INT   DEFAULT 0    -- how many times this entry was replayed
  stale_hits      INT   DEFAULT 0    -- how many times staleness was detected
  created_at      TEXT               -- first cached
  last_hit        TEXT               -- most recent valid cache hit
  last_stale      TEXT               -- most recent stale detection

Staleness
─────────
When a cache entry is written we snapshot the current device topology:
  { "rooms": ["bedroom", "bathroom", ...], "doors": ["front", "back", ...] }

On every cache lookup we compare the stored snapshot against the live
home_state topology.  If any room or door has been added or removed the
entry is considered STALE and None is returned — the model will handle
the request fresh and write a new, correct entry.

Adding a new light or door automatically invalidates every cache entry
that touched those device types, with no manual intervention needed.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path("cache.db")


# ── Lifecycle ──────────────────────────────────────────────────────────────────

def init_db(path: Path = _DB_PATH) -> None:
    """Create / migrate the tool_cache table."""
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS tool_cache (
                message         TEXT    PRIMARY KEY,
                tool_calls      TEXT    NOT NULL,
                device_snapshot TEXT    NOT NULL DEFAULT '{}',
                hits            INTEGER NOT NULL DEFAULT 0,
                stale_hits      INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    NOT NULL,
                last_hit        TEXT,
                last_stale      TEXT
            )
        """)
        # ── Migrate existing DBs that pre-date device_snapshot ─────────────
        cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(tool_cache)").fetchall()
        }
        for col, definition in [
            ("device_snapshot", "TEXT NOT NULL DEFAULT '{}'"),
            ("stale_hits",      "INTEGER NOT NULL DEFAULT 0"),
            ("last_stale",      "TEXT"),
        ]:
            if col not in cols:
                con.execute(f"ALTER TABLE tool_cache ADD COLUMN {col} {definition}")


# ── Device snapshot helpers ────────────────────────────────────────────────────

def build_snapshot(home_state: dict) -> dict:
    """
    Extract the device topology (just the keys, not values) from home_state.
    We care only whether the SET of devices has changed, not their on/off status.
    """
    return {
        "rooms": sorted(home_state.get("lights", {}).keys()),
        "doors": sorted(home_state.get("doors",  {}).keys()),
    }


def _snapshot_matches(stored_raw: str, current: dict) -> bool:
    """
    True when stored topology == current topology.
    A missing / empty stored snapshot is treated as a mismatch so that entries
    written before this feature was introduced are gracefully invalidated.
    """
    if not stored_raw or stored_raw == "{}":
        return False
    try:
        stored = json.loads(stored_raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        sorted(stored.get("rooms", [])) == sorted(current.get("rooms", []))
        and
        sorted(stored.get("doors", [])) == sorted(current.get("doors", []))
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def get_cached(
    message: str,
    current_snapshot: dict,
    path: Path = _DB_PATH,
) -> list[dict] | None:
    """
    Return cached tool calls for *message*, or None on miss / stale.

    current_snapshot  →  build_snapshot(home_state) from the caller.

    Cache HIT   → bumps hits + last_hit,         returns tool_calls list.
    STALE HIT   → bumps stale_hits + last_stale, returns None (fall through to model).
    MISS        → returns None.
    """
    key = _normalise(message)
    with _connect(path) as con:
        row = con.execute(
            "SELECT tool_calls, device_snapshot FROM tool_cache WHERE message = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        if not _snapshot_matches(row["device_snapshot"], current_snapshot):
            con.execute(
                """UPDATE tool_cache
                   SET stale_hits = stale_hits + 1, last_stale = ?
                   WHERE message = ?""",
                (_now(), key),
            )
            return None

        con.execute(
            "UPDATE tool_cache SET hits = hits + 1, last_hit = ? WHERE message = ?",
            (_now(), key),
        )
    return json.loads(row["tool_calls"])


def set_cached(
    message: str,
    tool_calls: list[dict],
    current_snapshot: dict,
    path: Path = _DB_PATH,
) -> None:
    """
    Store tool_calls for message, recording the current device topology.
    Replaces any existing entry for the same message.
    Never call this for text-only turns or failed actions.
    """
    if not tool_calls:
        return
    key      = _normalise(message)
    calls_js = json.dumps(tool_calls)
    snap_js  = json.dumps(current_snapshot)
    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO tool_cache
                (message, tool_calls, device_snapshot, hits, stale_hits, created_at, last_hit, last_stale)
            VALUES (?, ?, ?, 0, 0, ?, NULL, NULL)
            ON CONFLICT(message) DO UPDATE SET
                tool_calls      = excluded.tool_calls,
                device_snapshot = excluded.device_snapshot,
                created_at      = excluded.created_at,
                hits            = 0,
                stale_hits      = 0,
                last_hit        = NULL,
                last_stale      = NULL
            """,
            (key, calls_js, snap_js, _now()),
        )


def delete_cached(message: str, path: Path = _DB_PATH) -> bool:
    """Remove a single entry by message text.  Returns True if deleted."""
    key = _normalise(message)
    with _connect(path) as con:
        return con.execute(
            "DELETE FROM tool_cache WHERE message = ?", (key,)
        ).rowcount > 0


def clear_all(path: Path = _DB_PATH) -> int:
    """Wipe the entire cache.  Returns number of rows deleted."""
    with _connect(path) as con:
        return con.execute("DELETE FROM tool_cache").rowcount


def clear_stale(current_snapshot: dict, path: Path = _DB_PATH) -> int:
    """
    Delete every entry whose stored snapshot no longer matches current_snapshot.
    Returns number of rows deleted.
    """
    entries = list_entries(current_snapshot=current_snapshot, path=path)
    stale_keys = [
        _normalise(e["message"]) for e in entries if e["is_stale"]
    ]
    if not stale_keys:
        return 0
    with _connect(path) as con:
        return con.execute(
            f"DELETE FROM tool_cache WHERE message IN ({','.join('?'*len(stale_keys))})",
            stale_keys,
        ).rowcount


def list_entries(
    current_snapshot: dict | None = None,
    path: Path = _DB_PATH,
) -> list[dict]:
    """
    Return all cache entries ordered newest-first.
    When current_snapshot is provided each entry gains an `is_stale` bool.
    """
    with _connect(path) as con:
        rows = con.execute(
            """SELECT message, tool_calls, device_snapshot,
                      hits, stale_hits, created_at, last_hit, last_stale
               FROM tool_cache
               ORDER BY created_at DESC"""
        ).fetchall()

    result = []
    for r in rows:
        snap_raw = r["device_snapshot"] or "{}"
        entry: dict = {
            "message":         r["message"],
            "tool_calls":      json.loads(r["tool_calls"]),
            "device_snapshot": json.loads(snap_raw) if snap_raw != "{}" else {},
            "hits":            r["hits"],
            "stale_hits":      r["stale_hits"],
            "created_at":      r["created_at"],
            "last_hit":        r["last_hit"],
            "last_stale":      r["last_stale"],
        }
        if current_snapshot is not None:
            entry["is_stale"] = not _snapshot_matches(snap_raw, current_snapshot)
        result.append(entry)
    return result


# ── Internal helpers ───────────────────────────────────────────────────────────

def _normalise(message: str) -> str:
    return " ".join(message.lower().split())


def _connect(path: Path = _DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
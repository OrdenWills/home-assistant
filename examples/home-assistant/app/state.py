"""
app/state.py
────────────
home_state is the single source of truth for the home's current condition.
It is persisted to SQLite (the same cache.db used by app/cache.py) so that
state survives server restarts.

Call load_state() once at startup to restore the last-saved snapshot.
Every tool handler mutates home_state directly; the handlers call
persist_state() at the end of each mutation so the DB stays in sync.
"""

import json
import random
import sqlite3
from pathlib import Path

_DB_PATH = Path("cache.db")

# ── In-memory state (canonical runtime copy) ──────────────────────────────────

home_state: dict = {
    "lights": {
        "living_room": {"state": "off"},
        "bedroom":     {"state": "off"},
        "kitchen":     {"state": "off"},
        "bathroom":    {"state": "off"},
        "office":      {"state": "off"},
        "hallway":     {"state": "off"},
    },
    "thermostat": {"temperature": 68, "mode": "auto"},
    "doors": {
        "front":  "locked",
        "back":   "locked",
        "garage": "locked",
        "side":   "locked",
        "bedroom":     "locked",
        "bathroom":    "locked",
        "office":      "locked",
        "kitchen":     "locked",
        "living_room": "locked",
    },
    "active_scene": None,
}


# ── Persistence helpers ────────────────────────────────────────────────────────

def _connect(path: Path = _DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def init_state_table(path: Path = _DB_PATH) -> None:
    """Create the home_state table if it does not exist."""
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS home_state (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT    NOT NULL
            )
        """)


def load_state(path: Path = _DB_PATH) -> None:
    """
    Overwrite home_state with the snapshot stored in the DB.
    If no snapshot exists the default values remain in place.
    """
    init_state_table(path)
    with _connect(path) as con:
        row = con.execute("SELECT payload FROM home_state WHERE id = 1").fetchone()
    if row is None:
        print("[State] No persisted state found — using defaults.")
        return
    try:
        stored = json.loads(row["payload"])
        
        # Deep merge for structured device dictionaries to preserve new keys
        for key in ["lights", "doors"]:
            if key in stored and isinstance(stored[key], dict):
                home_state[key].update(stored[key])
        
        # Shallow update for other top-level keys
        for key, value in stored.items():
            if key not in ["lights", "doors"]:
                home_state[key] = value
                
        print("[State] Restored persisted home state from DB (merged).")
    except Exception as e:
        print(f"[State] Could not restore state ({e}) — using defaults.")


def persist_state(path: Path = _DB_PATH) -> None:
    """
    Write the current home_state snapshot to the DB.
    Called automatically by every tool handler that mutates state.
    """
    payload = json.dumps(home_state)
    with _connect(path) as con:
        con.execute("""
            INSERT INTO home_state (id, payload)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET payload = excluded.payload
        """, (payload,))


# ── Utility (used for training-data generation) ───────────────────────────────

def randomize_state() -> None:
    """Randomize lights and doors in home_state (does NOT persist)."""
    for room in home_state["lights"]:
        home_state["lights"][room]["state"] = random.choice(["on", "off"])
    for door in home_state["doors"]:
        home_state["doors"][door] = random.choice(["locked", "unlocked"])
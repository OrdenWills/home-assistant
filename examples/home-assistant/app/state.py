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
from datetime import datetime
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
    "tv": {
        "living_room": "off",
        # "bedroom": "off",
    },
    "speaker": {
        "living_room": "stopped",
        # "bedroom": "stopped",
        # "kitchen": "stopped",
        # "office": "stopped",
        # "hallway": "stopped",
    },
    "fan": {
        "living_room": {"state": "off", "speed": "medium"},
        "bedroom": {"state": "off", "speed": "medium"},
        "kitchen": {"state": "off", "speed": "medium"},
        "office": {"state": "off", "speed": "medium"},
    },
    "active_model_id": "home-assistant-sft(small)",
    "music_folder": None,
}


# ── Persistence helpers ────────────────────────────────────────────────────────

def _connect(path: Path = _DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def init_state_table(path: Path = _DB_PATH) -> None:
    """Create the home_state table if it does not exist."""
    with _connect(path) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS home_state (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER,
                payload TEXT    NOT NULL
            );
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
        for key in ["lights", "doors", "tv", "speaker", "fan"]:
            if key in stored and isinstance(stored[key], dict):
                home_state[key].update(stored[key])
        
        # Shallow update for other top-level keys
        for key, value in stored.items():
            if key not in ["lights", "doors", "tv", "speaker", "fan"]:
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


def load_chat_history(path: Path = _DB_PATH) -> list[dict]:
    """Load all chat turns from the DB."""
    init_state_table(path)
    with _connect(path) as con:
        rows = con.execute("SELECT payload FROM chat_history ORDER BY id ASC").fetchall()
    return [json.loads(row["payload"]) for row in rows]


def save_chat_history(turns: list[dict], path: Path = _DB_PATH) -> None:
    """Overwrite the chat_history table with the current list of turns."""
    init_state_table(path)
    with _connect(path) as con:
        con.execute("DELETE FROM chat_history")
        con.executemany(
            "INSERT INTO chat_history (turn_id, payload) VALUES (?, ?)",
            [(t.get("turn_id"), json.dumps(t)) for t in turns]
        )


def clear_chat_history(path: Path = _DB_PATH) -> None:
    """Wipe the chat history table."""
    with _connect(path) as con:
        con.execute("DELETE FROM chat_history")


# ── Utility (used for training-data generation) ───────────────────────────────

def randomize_state() -> None:
    """Randomize lights and doors in home_state (does NOT persist)."""
    for room in home_state["lights"]:
        home_state["lights"][room]["state"] = random.choice(["on", "off"])
    for door in home_state["doors"]:
        home_state["doors"][door] = random.choice(["locked", "unlocked"])


# ── Structured state summary (injected into every LLM request) ────────────────

def build_state_summary(current_room: str | None = None) -> str:
    """
    Build a compact structured state string for LLM injection.
    Includes user's current room position.

    Example output:
    [STATE: lights={bathroom:off, bedroom:on, ...}, doors={back:locked, ...},
     thermostat=72°F/auto, scene=none, user_room=bedroom]
    """
    lights = ", ".join(
        f"{r}:{d['state']}" for r, d in sorted(home_state["lights"].items())
    )
    doors = ", ".join(
        f"{d}:{s}" for d, s in sorted(home_state["doors"].items())
    )
    tv_str = ", ".join(f"{r}:{s}" for r, s in sorted(home_state.get("tv", {}).items())) if home_state.get("tv") else ""
    sp_str = ", ".join(f"{r}:{s}" for r, s in sorted(home_state.get("speaker", {}).items())) if home_state.get("speaker") else ""
    fan_str = ", ".join(f"{r}:{d['state']}({d.get('speed', 'medium')})" for r, d in sorted(home_state.get("fan", {}).items())) if home_state.get("fan") else ""
    therm = home_state["thermostat"]
    scene = home_state.get("active_scene") or "none"
    room = current_room or ""
    return (
        f"[STATE: lights={{{lights}}}, "
        f"doors={{{doors}}}, "
        f"thermostat={therm['temperature']}F/{therm['mode']}, "
        f"scene={scene}, "
        f"tv={{{tv_str}}}, "
        f"speaker={{{sp_str}}}, "
        f"fan={{{fan_str}}}, "
        f"current_user_room={room}]"
    )


# ── Action log (replaces raw conversation_history sliding window) ─────────────

action_log: list[dict] = []
MAX_ACTION_LOG = 3


def log_action(action_name: str, args: dict, summary: str) -> None:
    """Append an action to the log, keeping at most MAX_ACTION_LOG entries."""
    action_log.append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action_name,
        "args": args,
        "summary": summary,
    })
    if len(action_log) > MAX_ACTION_LOG:
        action_log[:] = action_log[-MAX_ACTION_LOG:]


def build_action_log_context(n: int = 3) -> str:
    """
    Return the last N actions as a compact context string.
    Returns empty string when no actions have been logged.
    """
    if not action_log:
        return ""
    recent = action_log[-n:]
    entries = "; ".join(
        f"{a['time']} {a['action']}({', '.join(f'{k}={v}' for k, v in a['args'].items())}) -> {a['summary']}"
        for a in recent
    )
    return f"[RECENT ACTIONS: {entries}]"
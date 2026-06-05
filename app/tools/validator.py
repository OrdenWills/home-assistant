# app/tools/validator.py
#
# Fix 1 — Argument coercion:   silently remaps wrong-key hallucinations before
#                               they ever reach a handler.
# Fix 2 — Pre-execution schema: validates every call against a known schema and
#                               returns a structured error string (not an
#                               exception) so the agent loop can feed it back
#                               to the model (Fix 3) instead of returning 500.

from __future__ import annotations
from typing import Any

# ── Allowed-value sets (shared across coercion + validation) ───────────────────

_ON_OFF  = {"on", "off"}
_LOCK_OP = {"lock", "unlock"}
_THERMO  = {"heat", "cool", "auto"}
_SCENES  = {"movie_night", "bedtime", "morning", "away", "party"}
_ROOMS   = {"bedroom", "bathroom", "office", "hallway", "kitchen", "living_room"}
_DOORS   = {"front", "back", "garage", "side", "bedroom", "bathroom", "office", "kitchen", "living_room"}
_TV_ROOMS = {"living_room", "bedroom"}
_SPEAKER_ROOMS = {"living_room", "bedroom", "kitchen", "office", "hallway"}
_SPEAKER_ACTIONS = {"play", "pause", "stop", "next", "previous", "volume"}
_FAN_ROOMS = {"living_room", "bedroom", "kitchen", "office"}
_FAN_SPEEDS = {"low", "medium", "high"}

# ── Required-param schema ──────────────────────────────────────────────────────
# Maps tool_name → {param: allowed_values_or_None}
# None means "any non-empty value accepted".

PARAM_SCHEMA: dict[str, dict[str, set | None]] = {
    "toggle_lights":     {"room": _ROOMS,  "state": _ON_OFF},
    "toggle_all_lights": {"state": _ON_OFF},
    "lock_door":         {"door": _DOORS,  "state": _LOCK_OP},
    "lock_all_doors":    {"state": _LOCK_OP},
    "set_thermostat":    {"temperature": None, "mode": _THERMO},
    "set_scene":         {"scene": _SCENES},
    "control_tv":        {"room": _TV_ROOMS, "state": _ON_OFF},
    "control_speaker":   {"room": _SPEAKER_ROOMS, "action": _SPEAKER_ACTIONS},
    "control_fan":       {"room": _FAN_ROOMS, "state": _ON_OFF},
    "intent_unclear":    {"reason": None},
}

# ── Fix 1: Argument coercion ───────────────────────────────────────────────────

def coerce_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Return a corrected copy of args. Never raises.

    Handles the hallucination class where the model uses the right function but
    puts a value under the wrong key, e.g.:
        toggle_all_lights(room="off")   →  toggle_all_lights(state="off")
        toggle_lights(action="on", ...) →  toggle_lights(state="on", ...)
        lock_door(door="lock", ...)     →  lock_door(state="lock", ...)

    Strategy: for each required param that is missing or has an invalid value,
    scan all other keys for a value that belongs to that param's allowed set
    and remap it.
    """
    schema = PARAM_SCHEMA.get(tool_name)
    if not schema:
        return args

    args = dict(args)  # never mutate the caller's dict

    for param, allowed in schema.items():
        if allowed is None:
            continue  # free-form param — nothing to coerce
        current_val = str(args.get(param, "")).lower()
        if current_val in allowed:
            continue  # already correct

        # Search other keys for a stray value that belongs here
        for key, val in list(args.items()):
            if key == param:
                continue
            if str(val).lower() in allowed:
                print(f"[Coerce] {tool_name}: '{key}'={val!r} remapped → '{param}'={val!r}")
                args[param] = str(val).lower()
                del args[key]
                break

    return args


# ── Fix 2: Pre-execution validation ───────────────────────────────────────────

def validate_tool_call(tool_name: str, args: dict[str, Any]) -> str | None:
    """
    Validate args against PARAM_SCHEMA.
    Returns None when everything is fine, or a short error string the agent
    can feed back to the model as the tool result so it can self-correct.
    """
    schema = PARAM_SCHEMA.get(tool_name)
    if schema is None:
        return f"Unknown tool '{tool_name}'."

    errors: list[str] = []

    for param, allowed in schema.items():
        if param not in args:
            hint = f"one of {sorted(allowed)}" if allowed else "a non-empty value"
            errors.append(f"missing required param '{param}' (expected {hint})")
            continue

        if allowed is not None:
            val = str(args[param]).lower()
            if val not in allowed:
                errors.append(
                    f"invalid value for '{param}': got {args[param]!r}, "
                    f"expected one of {sorted(allowed)}"
                )

    # Extra range check for thermostat temperature
    if tool_name == "set_thermostat" and "temperature" in args:
        try:
            t = float(args["temperature"])
            if not (60 <= t <= 80):
                errors.append(f"'temperature' must be 60–80°F, got {t}")
        except (TypeError, ValueError):
            errors.append(f"'temperature' must be a number, got {args['temperature']!r}")

    # Extra check for fan speed (optional)
    if tool_name == "control_fan" and "speed" in args:
        val = str(args["speed"]).lower()
        if val not in _FAN_SPEEDS:
            errors.append(f"invalid value for 'speed': got {args['speed']!r}, expected one of {sorted(_FAN_SPEEDS)}")

    if errors:
        return (
            f"Tool call error in {tool_name}(): "
            + "; ".join(errors)
            + ". Please correct the arguments and try again."
        )

    return None

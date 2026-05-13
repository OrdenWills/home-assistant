import os
import random
from app.state import home_state, persist_state

# ── Lazy pygame mixer init ─────────────────────────────────────────────────────
# pygame.mixer.init() is slow (~2-5 s) and blocks the event loop.
# We defer it until the first actual speaker action so server reloads are instant.
_mixer_ready = False

def _ensure_mixer():
    global _mixer_ready
    if _mixer_ready:
        return True
    try:
        import pygame
        pygame.mixer.init()
        _mixer_ready = True
        print("[Audio] pygame mixer initialised (lazy).")
        return True
    except Exception as e:
        print(f"[Audio] pygame mixer init failed: {e}")
        return False


def toggle_lights(room: str, state: str) -> dict:
    if room == "all":
        return toggle_all_lights(state)
    home_state["lights"][room]["state"] = state
    persist_state()
    return {"success": True, "room": room, "state": state}


def set_thermostat(temperature: int, mode: str) -> dict:
    home_state["thermostat"]["temperature"] = temperature
    home_state["thermostat"]["mode"] = mode
    persist_state()
    return {"success": True, "temperature": temperature, "mode": mode}


def lock_door(door: str, state: str) -> dict:
    if door == "all":
        return lock_all_doors(state)
    home_state["doors"][door] = "locked" if state == "lock" else "unlocked"
    persist_state()
    return {"success": True, "door": door, "state": home_state["doors"][door]}


def get_device_status(device_type: str, room: str = None) -> dict:
    # Read-only — no persist needed
    if device_type == "lights":
        if room:
            return {"device_type": "lights", "room": room, "status": home_state["lights"].get(room)}
        return {"device_type": "lights", "status": home_state["lights"]}
    elif device_type == "thermostat":
        return {"device_type": "thermostat", "status": home_state["thermostat"]}
    elif device_type == "door":
        if room:
            return {"device_type": "door", "door": room, "status": home_state["doors"].get(room)}
        return {"device_type": "door", "status": home_state["doors"]}
    else:  # "all"
        return {
            "lights":       home_state["lights"],
            "thermostat":   home_state["thermostat"],
            "doors":        home_state["doors"],
            "active_scene": home_state["active_scene"],
        }


def set_scene(scene: str) -> dict:
    lights = home_state["lights"]
    doors  = home_state["doors"]
    therm  = home_state["thermostat"]

    if scene == "movie_night":
        lights["living_room"]["state"] = "on"
        if "tv" in home_state and "living_room" in home_state["tv"]:
            home_state["tv"]["living_room"] = "on"
        if "speaker" in home_state and "living_room" in home_state["speaker"]:
            home_state["speaker"]["living_room"] = "playing"
        therm["temperature"] = 72
        therm["mode"] = "auto"
    elif scene == "bedtime":
        for room in lights:
            lights[room]["state"] = "off"
        for door in doors:
            doors[door] = "locked"
        for room in home_state.get("tv", {}):
            home_state["tv"][room] = "off"
        for room in home_state.get("speaker", {}):
            home_state["speaker"][room] = "stopped"
        therm["temperature"] = 68
        therm["mode"] = "auto"
    elif scene == "morning":
        lights["kitchen"]["state"] = "on"
        lights["hallway"]["state"] = "on"
        therm["temperature"] = 72
        therm["mode"] = "auto"
    elif scene == "away":
        for room in lights:
            lights[room]["state"] = "off"
        for door in doors:
            doors[door] = "locked"
        for room in home_state.get("tv", {}):
            home_state["tv"][room] = "off"
        for room in home_state.get("speaker", {}):
            home_state["speaker"][room] = "stopped"
        therm["temperature"] = 65
        therm["mode"] = "auto"
    elif scene == "party":
        lights["living_room"]["state"] = "on"
        lights["kitchen"]["state"] = "on"
        therm["temperature"] = 70
        therm["mode"] = "auto"
        if "speaker" in home_state and "living_room" in home_state["speaker"]:
            control_speaker("living_room", "play")

    home_state["active_scene"] = scene
    persist_state()
    return {"success": True, "scene": scene}


def intent_unclear(reason: str = "unknown") -> dict:
    # No state change — no persist needed
    return {"success": False, "reason": reason}


def toggle_all_lights(state: str) -> dict:
    for room in home_state["lights"]:
        home_state["lights"][room]["state"] = state
    persist_state()
    affected_rooms = list(home_state["lights"].keys())
    return {"success": True, "state": state, "rooms": affected_rooms, "count": len(affected_rooms)}


def lock_all_doors(state: str) -> dict:
    target_state = "locked" if state == "lock" else "unlocked"
    for door in home_state["doors"]:
        home_state["doors"][door] = target_state
    persist_state()
    affected_doors = list(home_state["doors"].keys())
    return {"success": True, "state": target_state, "doors": affected_doors, "count": len(affected_doors)}


def control_tv(room: str, state: str) -> dict:
    if "tv" not in home_state or room not in home_state["tv"]:
        return {"success": False, "error": f"No TV in {room}"}
    home_state["tv"][room] = state
    persist_state()
    return {"success": True, "room": room, "state": state}


def control_speaker(room: str, action: str, media: str = None) -> dict:
    if "speaker" not in home_state or room not in home_state["speaker"]:
        return {"success": False, "error": f"No speaker in {room}"}
    
    current_status = home_state["speaker"][room]
    
    if action == "play":
        home_state["speaker"][room] = "playing"
    elif action == "pause":
        home_state["speaker"][room] = "paused"
    elif action == "stop":
        home_state["speaker"][room] = "stopped"
    elif action in ["next", "previous"]:
        home_state["speaker"][room] = "playing"
    
    persist_state()
    
    res = {"success": True, "room": room, "action": action, "state": home_state["speaker"][room]}
    if media:
        res["media"] = media
    
    # Real audio playback logic
    music_folder = home_state.get("music_folder")
    if music_folder and os.path.exists(music_folder):
        try:
            # Filter for common audio files
            files = sorted([f for f in os.listdir(music_folder) if f.lower().endswith(('.mp3', '.wav', '.ogg'))])
            if files:
                res["library_count"] = len(files)
                idx = home_state.get("current_track_index", 0)
                
                # Boundary check
                if idx >= len(files): idx = 0
                
                # ── Fuzzy match when media query is provided ───────────────
                if media and action in ["play"]:
                    idx = _fuzzy_match_track(media, files, idx)
                    res["fuzzy_query"] = media
                elif action == "next":
                    idx = (idx + 1) % len(files)
                elif action == "previous":
                    idx = (idx - 1) % len(files)
                
                home_state["current_track_index"] = idx
                
                track_name = files[idx]
                track_path = os.path.join(music_folder, track_name)
                home_state["current_track_name"] = track_name
                persist_state()
                res["current_track"] = track_name
                
                # Perform actual audio actions
                if _ensure_mixer():
                    import pygame
                    if action in ["play", "next", "previous"]:
                        try:
                            pygame.mixer.music.load(track_path)
                            pygame.mixer.music.play()
                        except Exception as e:
                            res["playback_error"] = str(e)
                    elif action == "pause":
                        pygame.mixer.music.pause()
                    elif action == "stop":
                        pygame.mixer.music.stop()
                    elif action == "resume" or (action == "play" and current_status == "paused"):
                        pygame.mixer.music.unpause()
            else:
                res["info"] = "No supported audio files found in folder."
        except Exception as e:
            res["error"] = f"Folder error: {str(e)}"
            
    return res


# ── Fuzzy track matching ───────────────────────────────────────────────────────
import re as _re

_FUZZY_THRESHOLD = 50  # minimum score (0-100) to accept a match

# YouTube ID suffix: _<11 alphanum/dash chars> at end  (e.g. _shaX6kpDczI)
_YT_ID_RE = _re.compile(r'[_][A-Za-z0-9_-]{11}$')
# Website / source tags in brackets or parens
_SOURCE_TAG_RE = _re.compile(
    r'[\[\(]\s*(?:TrendyBeatz\.com|belaloaded\.com_?|okhype\.com|'
    r'talkglitz\.tv|www\.\S+)\s*[\]\)]?', _re.IGNORECASE)
# Common noise suffixes like (Official Audio), (Official Music Video)
_NOISE_RE = _re.compile(
    r'\(?\s*(?:Official\s*(?:Audio|Video|Music\s*Video)?|'
    r'Piano\s*Tutorial)\s*\)?', _re.IGNORECASE)
# Trailing filler the model appends to media queries
_QUERY_FILLER_RE = _re.compile(
    r'\b(?:for\s+me|for\s+us|please|on\s+the\s+speaker)\s*$', _re.IGNORECASE)
# Leading article (only stripped when query has more substance)
_LEADING_ARTICLE_RE = _re.compile(r'^(?:the|a|an)\s+', _re.IGNORECASE)


def _clean_filename(name: str) -> str:
    """Strip noise from a filename (no extension) for better matching."""
    name = _SOURCE_TAG_RE.sub('', name)
    name = _NOISE_RE.sub('', name)
    name = _YT_ID_RE.sub('', name)
    name = name.replace('-', ' ').replace('_', ' ')
    return _re.sub(r'\s+', ' ', name).strip()


def _clean_query(query: str) -> str:
    """Strip trailing filler and leading articles from a media query."""
    q = _QUERY_FILLER_RE.sub('', query).strip()
    stripped = _LEADING_ARTICLE_RE.sub('', q)
    if len(stripped) >= 2:
        q = stripped
    return q.strip()


def _fuzzy_match_track(query: str, files: list[str], fallback_idx: int) -> int:
    """
    Score *query* against every filename using rapidfuzz WRatio.
    
    Two-pass scoring for robustness:
      1. raw query   vs  cleaned filename
      2. core query  vs  cleaned filename  (filler words stripped)
    
    The *core* score acts as tiebreaker — e.g. "Lsd For Me" raw-matches
    both LSD tracks and "Love Me" equally at ~85, but the core "lsd" vs
    cleaned names clearly separates them (60 vs 30).
    
    Returns the index of the best match above _FUZZY_THRESHOLD.
    If several files tie, one is chosen at random.
    Falls back to *fallback_idx* when nothing matches well enough.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        print("[Audio] rapidfuzz not installed — skipping fuzzy match.")
        return fallback_idx

    q_lower = query.lower().strip()
    q_core = _clean_query(query).lower()

    scored: list[tuple[int, float]] = []  # (index, composite_score)

    for i, fname in enumerate(files):
        name_clean = _clean_filename(os.path.splitext(fname)[0]).lower()
        # Score raw query against cleaned filename
        raw_score = fuzz.WRatio(q_lower, name_clean)
        # Score core query (filler stripped) against cleaned filename
        core_score = fuzz.WRatio(q_core, name_clean)
        # Composite: whichever is higher wins, but core gets a small
        # bonus (+1) as tiebreaker since it's the truer intent signal
        best = max(raw_score, core_score + 1)
        if best >= _FUZZY_THRESHOLD:
            scored.append((i, best))

    if not scored:
        print(f"[Audio] No fuzzy match above {_FUZZY_THRESHOLD}% for: {query!r}")
        return fallback_idx

    best_score = max(s for _, s in scored)
    top_matches = [i for i, s in scored if s == best_score]
    winner = random.choice(top_matches)

    match_name = files[winner]
    print(f"[Audio] Fuzzy match: {query!r} -> {match_name!r} "
          f"(score={best_score:.0f}, candidates={len(top_matches)})")
    return winner


def control_fan(room: str, state: str, speed: str = None) -> dict:
    if "fan" not in home_state or room not in home_state["fan"]:
        return {"success": False, "error": f"No fan in {room}"}
    home_state["fan"][room]["state"] = state
    if speed:
        home_state["fan"][room]["speed"] = speed
    persist_state()
    return {"success": True, "room": room, "state": state, "speed": home_state["fan"][room].get("speed")}


TOOL_HANDLERS = {
    "toggle_lights":     toggle_lights,
    "set_thermostat":    set_thermostat,
    "lock_door":         lock_door,
    "get_device_status": get_device_status,
    "set_scene":         set_scene,
    "toggle_all_lights": toggle_all_lights,
    "lock_all_doors":    lock_all_doors,
    "intent_unclear":    intent_unclear,
    "control_tv":        control_tv,
    "control_speaker":   control_speaker,
    "control_fan":       control_fan,
}

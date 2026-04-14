#app/server.py
import re
import subprocess
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from app.agent import local_client, run_agent, run_agent_stream
from app.state import home_state, load_state
from app.cache import init_db, get_cached, set_cached, delete_cached, clear_all, list_entries, build_snapshot
from app.tools.handlers import TOOL_HANDLERS
# ── Model registry ─────────────────────────────────────────────────────────────

LOCAL_MODELS = [
    {"id":"home-assistant-sft","name":"Home Assistant SFT (Finetuned)","hf_repo":"OrdenWills/LFM2.5-1.2B-home-assistant-sft","hf_file":"LFM2.5-1.2B-Instruct.Q4_K_M.gguf","size_label":"714 MB","score_label":"99%"},
    {"id":"home-assistant-sft(small)","name":"Home(small) Assistant SFT (Finetuned)","hf_repo":"OrdenWills/LFM2.5-350M-home-assistant-sft","hf_file":"LFM2.5-350M.Q4_K_M.gguf","size_label":"218 MB","score_label":"98%"},
    {"id":"lfm2-350m-q4","name":"LFM2-350M-Q4_0.gguf","hf_repo":"LiquidAI/LFM2-350M-GGUF","hf_file":"LFM2-350M-Q4_0.gguf","size_label":"209 MB","score_label":"45%"},
    {"id":"lfm25-1b-thinking-q4","name":"LFM2.5-1.2B-Thinking-Q4_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Thinking-GGUF","hf_file":"LFM2.5-1.2B-Thinking-Q4_0.gguf","size_label":"718 MB","score_label":"75%"},
    {"id":"lfm25-1b-thinking-q8","name":"LFM2.5-1.2B-Thinking-Q8_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Thinking-GGUF","hf_file":"LFM2.5-1.2B-Thinking-Q8_0.gguf","size_label":"1.28 GB","score_label":"82%"},
    {"id":"lfm25-1b-q4","name":"LFM2.5-1.2B-Instruct-Q4_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Instruct-GGUF","hf_file":"LFM2.5-1.2B-Instruct-Q4_0.gguf","size_label":"696 MB","score_label":"68%"},
    {"id":"lfm25-1b-q8","name":"LFM2.5-1.2B-Instruct-Q8_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Instruct-GGUF","hf_file":"LFM2.5-1.2B-Instruct-Q8_0.gguf","size_label":"1.25 GB","score_label":"53%"},
    {"id":"lfm2-vl-450m-q4","name":"LFM2-VL-450M-Q4_0.gguf","hf_repo":"LiquidAI/LFM2-VL-450M-GGUF","hf_file":"LFM2-VL-450M-Q4_0.gguf","size_label":"209 MB","score_label":"40%"},
]

# ── Module-level state ─────────────────────────────────────────────────────────

conversation_history: list[dict] = []
# Stores complete turns for history replay: {user, assistant, tool_calls}
chat_turns: list[dict] = []
active_backend: str = "local"

llama_proc: subprocess.Popen | None = None
llama_status: str = "idle"
llama_active_model_id: str | None = None
llama_error: str | None = None

# ── Room-injection classifier ──────────────────────────────────────────────────

_ROOM_NAMES: set[str] = {
    "living room", "living_room",
    "bedroom", "kitchen", "bathroom", "office", "hallway",
}

_SPECIFIC_DEVICES: set[str] = {
    "front door", "back door", "garage door", "side door",
    "bedroom door", "bathroom door", "office door", "kitchen door", "living room door",
    "front", "back", "garage", "side",
    "thermostat",
    "all lights", "all doors", "all rooms",
}

_BULK_KEYWORDS: set[str] = {
    "all ", "entire", "every room", "everywhere", "whole house",
    "the lights",
}

_LIGHT_DEVICE_WORDS: set[str] = {
    "light", "lights", "lamp", "lamps", "bulb", "bulbs",
}

# Relative/comparative words that require the model to know live light state
_RELATIVE_KEYWORDS: set[str] = {
    "other", "the other", "every other", "all other",
    "the rest", "rest of", "remaining", "except", "apart from",
    "others", "the others",
}


_DOOR_DEVICE_WORDS = {"door", "doors"}
_DOOR_THIS_WORDS   = {"this door", "the door", "this one", "it"}

def _room_hint(room: str) -> str:
    room_display = room.replace("_", " ")
    return (
        f"Room context: {room}. "
        f"If they ask to control 'the light', assume they mean the {room_display}. "
        f"If they ask to control 'the door' or 'this door', assume they mean the {room_display} door."
    )

def _should_inject_room(message: str) -> bool:
    lower = message.lower()
    if any(r in lower for r in _ROOM_NAMES):
        return False
    if any(d in lower for d in _SPECIFIC_DEVICES):
        return False
    if any(b in lower for b in _BULK_KEYWORDS):
        return False
    # Inject for light OR ambiguous door commands
    has_light = any(w in lower for w in _LIGHT_DEVICE_WORDS)
    has_door  = any(w in lower for w in _DOOR_THIS_WORDS)
    return has_light or has_door

def _has_relative_keyword(message: str) -> bool:
    lower = message.lower()
    return any(k in lower for k in _RELATIVE_KEYWORDS)


def _lights_state_note(state: dict, current_room: str | None = None) -> str:
    """
    Build a concise lights-state string for injection into the system note.
    e.g. 'Lights ON: bedroom, kitchen. Lights OFF: bathroom, hallway, office, living_room.'
    Optionally flags which room the user is currently in so the model can
    resolve 'other' / 'the rest' correctly.
    """
    lights = state.get("lights", {})
    on_rooms  = sorted(r for r, v in lights.items() if v.get("state") == "on")
    off_rooms = sorted(r for r, v in lights.items() if v.get("state") != "on")

    on_str  = ", ".join(on_rooms)  or "none"
    off_str = ", ".join(off_rooms) or "none"

    note = f"Lights ON: {on_str}. Lights OFF: {off_str}."
    if current_room:
        note += f" The user is in the {current_room}."
    return note


# ── Background thread helper ───────────────────────────────────────────────────

def _start_llama_server_bg(model: dict) -> None:
    global llama_proc, llama_status, llama_active_model_id, llama_error, active_backend
    llama_status = "starting"
    llama_active_model_id = model["id"]
    llama_error = None

    if llama_proc is not None:
        try:
            llama_proc.terminate(); llama_proc.wait(timeout=10)
        except Exception:
            llama_proc.kill()
        llama_proc = None

    print(f"[LlamaServer] Starting: {model['name']}")
    model_path = None
    try:
        model_path = hf_hub_download(repo_id=model["hf_repo"], filename=model["hf_file"], repo_type="model", local_files_only=True)
        print(f"[LlamaServer] Using cached model: {model_path}")
    except Exception:
        print("[LlamaServer] Model not in cache, will download on demand...")

    base_cmd = "C:\\Users\\Adolphus\\llama-b8479-bin-win-cpu-x64\\llama-server.exe"
    if model_path:
        cmd = [base_cmd, "--model", model_path, "--port", "8080", "--ctx-size", "4096", "--n-gpu-layers", "99"]
    else:
        cmd = [base_cmd, "--hf-repo", model["hf_repo"], "--hf-file", model["hf_file"], "--port", "8080", "--ctx-size", "4096", "--n-gpu-layers", "99"]

    print(f"[LlamaServer] Command: {' '.join(cmd)}")
    try:
        llama_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        import threading as th
        def log_output():
            try:
                for line in iter(llama_proc.stdout.readline, ""):
                    if line: print(f"[LlamaServer] {line.rstrip()}")
            except Exception: pass
        th.Thread(target=log_output, daemon=True).start()
    except Exception as e:
        llama_status = "error"; llama_error = str(e)
        print(f"[LlamaServer] Error starting process: {e}"); return

    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8080/v1/models", timeout=2)
            llama_status = "ready"; active_backend = "local"
            print(f"[LlamaServer] Ready! Loaded in {time.time()-(deadline-1800):.1f}s"); return
        except Exception:
            time.sleep(2)

    llama_status = "error"
    llama_error = "llama-server did not become ready within 1800s"
    print(f"[LlamaServer] {llama_error}")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app_: FastAPI):
    init_db()
    print("[Cache] SQLite tool-call cache initialised.")
    load_state()

    default_model_id = "home-assistant-sft(small)"
    model = next((m for m in LOCAL_MODELS if m["id"] == default_model_id), None)
    if model:
        print(f"[Startup] Auto-loading default model: {model['name']}")
        threading.Thread(target=_start_llama_server_bg, args=(model,), daemon=True).start()
    yield

    global llama_proc
    if llama_proc is not None:
        try:
            llama_proc.terminate(); llama_proc.wait(timeout=10)
        except Exception:
            llama_proc.kill()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse("index.html")

@app.get("/model")
def get_model():
    if active_backend == "openai":
        return JSONResponse({"name": "gpt-4o-mini"})
    if llama_active_model_id:
        model = next((m for m in LOCAL_MODELS if m["id"] == llama_active_model_id), None)
        if model: return JSONResponse({"name": model["name"]})
    return JSONResponse({"name": "No model loaded"})

@app.get("/backend")
def get_backend():
    return JSONResponse({"backend": active_backend})

class BackendRequest(BaseModel):
    backend: str

@app.post("/backend")
def set_backend(req: BackendRequest):
    global active_backend
    if req.backend not in ("local", "openai"):
        return JSONResponse({"error": "invalid backend"}, status_code=400)
    active_backend = req.backend
    conversation_history.clear()
    return JSONResponse({"backend": active_backend})

@app.get("/state")
def get_state():
    return JSONResponse(home_state)

class ThermostatRequest(BaseModel):
    temperature: int
    mode: str | None = "auto"

@app.post("/thermostat")
def set_thermostat_direct(req: ThermostatRequest):
    home_state["thermostat"]["temperature"] = req.temperature
    if req.mode:
        home_state["thermostat"]["mode"] = req.mode
    from app.state import persist_state
    persist_state()
    return JSONResponse({"status": "ok", "state": home_state})

@app.get("/history")
def get_history():
    """Returns the chat history so the frontend can survive a page refresh."""
    return JSONResponse({"history": chat_turns})

@app.post("/reset")
def reset():
    conversation_history.clear()
    chat_turns.clear()
    return JSONResponse({"ok": True})

@app.get("/local-models")
def get_local_models():
    return JSONResponse(LOCAL_MODELS)

@app.get("/local-model-status")
def get_local_model_status():
    return JSONResponse({"status": llama_status, "model_id": llama_active_model_id, "error": llama_error})

class LocalModelRequest(BaseModel):
    model_id: str

@app.post("/local-model")
def start_local_model(req: LocalModelRequest):
    model = next((m for m in LOCAL_MODELS if m["id"] == req.model_id), None)
    if model is None: return JSONResponse({"error": "unknown model_id"}, status_code=400)
    threading.Thread(target=_start_llama_server_bg, args=(model,), daemon=True).start()
    return JSONResponse({"status": "starting"})

@app.delete("/local-model")
def stop_local_model():
    global llama_proc, llama_status, llama_active_model_id, active_backend
    if llama_proc is not None:
        try: llama_proc.terminate(); llama_proc.wait(timeout=10)
        except Exception: llama_proc.kill()
        llama_proc = None
    llama_status = "idle"; llama_active_model_id = None; active_backend = "openai"
    return JSONResponse({"status": "idle"})

# ── Cache endpoints ────────────────────────────────────────────────────────────

@app.get("/cache")
def list_cache():
    return JSONResponse(list_entries())

@app.delete("/cache")
def wipe_cache():
    return JSONResponse({"deleted": clear_all()})

class CacheDeleteRequest(BaseModel):
    message: str

@app.delete("/cache/entry")
def delete_cache_entry(req: CacheDeleteRequest):
    return JSONResponse({"removed": delete_cached(req.message)})

# ── Cache replay helpers ───────────────────────────────────────────────────────

def _replay_cached_tools(cached_calls: list[dict], on_tool_call=None) -> tuple[list[dict], str]:
    events: list[dict] = []
    summaries: list[str] = []
    for tc in cached_calls:
        name = tc["name"]; args = tc.get("args", {})
        handler = TOOL_HANDLERS.get(name)
        result = handler(**args) if handler else {"error": f"Unknown tool: {name}"}
        events.append({"name": name, "args": args, "result": result})
        if on_tool_call: on_tool_call(name, args, result)
        summaries.append(_summarise_tool(name, args, result))
    return events, " ".join(summaries) if summaries else "Done."

def _summarise_tool(name: str, args: dict, result: dict) -> str:
    if not result.get("success"): return f"({name} failed)"
    if name == "toggle_lights":    return f"{args['room'].replace('_',' ').title()} light turned {args['state']}."
    if name == "toggle_all_lights":return f"All lights turned {args['state']}."
    if name == "lock_door":
        a = "locked" if args["state"]=="lock" else "unlocked"
        return f"{args['door'].title()} door {a}."
    if name == "lock_all_doors":
        a = "locked" if args["state"]=="lock" else "unlocked"
        return f"All doors {a}."
    if name == "set_thermostat":   return f"Thermostat set to {args['temperature']}°F in {args['mode']} mode."
    if name == "set_scene":        return f"{args['scene'].replace('_',' ').title()} scene activated."
    if name == "intent_unclear":   return f"Intent unclear ({args.get('reason','?')})."
    return "Done."

# ── Chat endpoint ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    current_room: str | None = None


@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Gate
    if active_backend == "local" and llama_status != "ready":
        msg = {
            "starting": "Model is still loading, please wait.",
            "idle":     "No local model loaded. Select a model from the LFM Local dropdown.",
            "error":    f"Local model failed to start: {llama_error}",
        }.get(llama_status, "Local model is not ready.")
        return JSONResponse({"text": msg, "tool_calls": []}, status_code=503)

    # 2. Context injection — room + live lights state when needed
    resolved_message = req.message
    room_injected = False
    is_relative = _has_relative_keyword(req.message)
    needs_room  = req.current_room and _should_inject_room(req.message)

    if needs_room or is_relative:
        parts: list[str] = []

        if is_relative:
            msg_lower = req.message.lower()
            target_state = "off" if any(w in msg_lower for w in ("off", "disable")) else \
                           "on"  if any(w in msg_lower for w in ("on", "enable"))  else None

            if target_state:
                candidate_rooms = [
                    r for r, data in home_state.get("lights", {}).items()
                    if data.get("state") != target_state and r != req.current_room
                ]
                if candidate_rooms:
                    # ── SHORT-CIRCUIT: execute directly, no model needed ──────────
                    events = []
                    summaries = []
                    for room in candidate_rooms:
                        args   = {"room": room, "state": target_state}
                        result = TOOL_HANDLERS["toggle_lights"](**args)
                        events.append({"name": "toggle_lights", "args": args, "result": result})
                        summaries.append(f"{room.replace('_', ' ').title()} light turned {target_state}.")
                        print(f"[ShortCircuit] toggle_lights(room={room!r}, state={target_state!r}) → {result}")

                    text = " ".join(summaries)
                    chat_turns.append({"user": req.message, "assistant": text, "tool_calls": events})

                    # Cache the pre-calculated calls
                    current_snapshot = build_snapshot(home_state)
                    to_cache = [{"name": e["name"], "args": e["args"]} for e in events]
                    set_cached(resolved_message, to_cache, current_snapshot)

                    return JSONResponse({
                        "text": text,
                        "tool_calls": events,
                        "cached": False,
                        "room_injected": True,
                    })
                else:
                    # Nothing to do
                    text = f"The other lights are already {target_state}."
                    chat_turns.append({"user": req.message, "assistant": text, "tool_calls": []})
                    return JSONResponse({"text": text, "tool_calls": [], "cached": False, "room_injected": True})
            else:
                parts.append(_lights_state_note(home_state, req.current_room))
                parts.append(
                    "Resolve 'other' or 'the rest' against the live state provided, "
                    "excluding the user's current room. Issue one toggle_lights call per room."
                )

        elif needs_room:
            parts.append(_room_hint(req.current_room))

        if parts:
            note = " ".join(parts)
            # Use the [HINT: ...] format! The model obeys this much better.
            resolved_message = f"{req.message}\n[HINT: {note}]"
            room_injected = True
            print(f"[RoomContext] Injected (room={needs_room}, relative={is_relative}) → {resolved_message!r}")
    else:
        print(f"[RoomContext] No injection — room={req.current_room!r}")

    # 3. Cache lookup (keyed on resolved_message so bedroom/kitchen are separate)
    current_snapshot = build_snapshot(home_state)
    cached = get_cached(resolved_message, current_snapshot)
    if cached:
        print(f"[Cache] HIT for: {resolved_message!r}")
        events, text = _replay_cached_tools(cached)
        chat_turns.append({"user": req.message, "assistant": text, "tool_calls": events})
        return JSONResponse({"text": text, "tool_calls": events, "cached": True, "room_injected": room_injected})

    # 4. Cache miss — invoke model
    print(f"[Cache] MISS for: {resolved_message!r} — invoking model")
    events: list[dict] = []

    def on_tool_call(name, args, result):
        print(f"[Server] Tool: {name}({args}) → {result}")
        events.append({"name": name, "args": args, "result": result})

    initial_history_len = len(conversation_history)
    messages_out: list[dict] = []
    try:
        text = run_agent(
            resolved_message,
            history=conversation_history,
            backend=active_backend,
            on_tool_call=on_tool_call,
            messages_out=messages_out,
        )
    except Exception as e:
        return JSONResponse({"text": f"Error: {e}", "tool_calls": events}, status_code=500)

    conversation_history.extend(messages_out[1 + initial_history_len:])

    # Keep only the last 2 turns (4 messages: 2 user + 2 assistant) for aggressive cleanup
    if len(conversation_history) > 4:
        conversation_history[:] = conversation_history[-4:]

    # Record this turn for history replay
    chat_turns.append({"user": req.message, "assistant": text, "tool_calls": events})

    # 5. Cache only successful, non-unclear tool calls
    action_events = [
        e for e in events
        if e["name"] != "intent_unclear" and e.get("result", {}).get("success")
    ]
    if action_events:
        to_cache = [{"name": e["name"], "args": e["args"]} for e in action_events]
        set_cached(resolved_message, to_cache, current_snapshot)
        print(f"[Cache] STORED {len(to_cache)} call(s) for: {resolved_message!r}")

    return JSONResponse({"text": text, "tool_calls": events, "cached": False, "room_injected": room_injected})

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE endpoint — streams tool_call events then text tokens."""

    # ── Gate ──────────────────────────────────────────────────────────────────
    if active_backend == "local" and llama_status != "ready":
        msg = {
            "starting": "Model is still loading, please wait.",
            "idle":     "No local model loaded. Select a model from the LFM Local dropdown.",
            "error":    f"Local model failed to start: {llama_error}",
        }.get(llama_status, "Local model is not ready.")

        def _gate_error():
            yield f"data: {json.dumps({'type': 'error', 'text': msg})}\n\n"
        return StreamingResponse(_gate_error(), media_type="text/event-stream")

    # ── Context injection (identical logic to /chat) ──────────────────────────
    resolved_message = req.message
    is_relative  = _has_relative_keyword(req.message)
    needs_room   = req.current_room and _should_inject_room(req.message)
    room_injected = False

    _sc_target = None
    _sc_rooms  = None
    if is_relative:
        msg_lower = req.message.lower()
        _sc_target = ("off" if any(w in msg_lower for w in ("off","disable")) else
                      "on"  if any(w in msg_lower for w in ("on","enable"))   else None)
        if _sc_target:
            _sc_rooms = [
                r for r, data in home_state.get("lights", {}).items()
                if data.get("state") != _sc_target and r != req.current_room
            ]

    if needs_room or is_relative:
        parts: list[str] = []
        if is_relative:
            if _sc_target:
                if _sc_rooms:
                    calls = " ".join(
                        f'[toggle_lights(room="{r}", state="{_sc_target}")]'
                        for r in _sc_rooms
                    )
                    parts.append(f"Target rooms pre-calculated. You MUST output exactly these calls: {calls}")
                else:
                    parts.append(f"The 'other' lights are already {_sc_target}. Respond in plain text saying no changes are needed.")
            else:
                parts.append(_lights_state_note(home_state, req.current_room))
                parts.append(
                    "Resolve 'other' or 'the rest' against the live state provided, "
                    "excluding the user's current room. Issue one toggle_lights call per room."
                )
        elif needs_room:
            parts.append(_room_hint(req.current_room))
        if parts:
            resolved_message = f"{req.message}\n[HINT: {' '.join(parts)}]"
            room_injected = True

    # ── Cache check ───────────────────────────────────────────────────────────
    current_snapshot = build_snapshot(home_state)
    cached = get_cached(resolved_message, current_snapshot)

    def generate():
        if is_relative and _sc_rooms is not None:
            target = _sc_target
            if _sc_rooms:
                events = []
                summaries = []
                for room in _sc_rooms:
                    yield f"data: {json.dumps({'type': 'status', 'text': 'Performing action...'})}\n\n"
                    args   = {"room": room, "state": target}
                    result = TOOL_HANDLERS["toggle_lights"](**args)
                    events.append({"name": "toggle_lights", "args": args, "result": result})
                    payload = {"type": "tool_call", "name": "toggle_lights",
                               "args": args, "result": result, "state": home_state}
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield f"data: {json.dumps({'type': 'status', 'text': 'Action performed...'})}\n\n"
                    summaries.append(f"{room.replace('_', ' ').title()} light turned {target}.")

                text = " ".join(summaries)
                chat_turns.append({"user": req.message, "assistant": text, "tool_calls": events})
                
                # Cache the pre-calculated calls
                to_cache = [{"name": e["name"], "args": e["args"]} for e in events]
                set_cached(resolved_message, to_cache, current_snapshot)
                
                yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'text': text, 'cached': False})}\n\n"
            else:
                text = f"The other lights are already {target}."
                chat_turns.append({"user": req.message, "assistant": text, "tool_calls": []})
                yield f"data: {json.dumps({'type': 'done', 'text': text, 'cached': False})}\n\n"
            return

        # ── Cache HIT: replay tools then emit done immediately ────────────────
        if cached:
            print(f"[Cache] HIT (stream) for: {resolved_message!r}")
            events, text = _replay_cached_tools(cached)
            for e in events:
                yield f"data: {json.dumps({'type': 'status', 'text': 'Performing action...'})}\n\n"
                payload = {"type": "tool_call", "name": e["name"],
                           "args": e["args"], "result": e["result"], "state": home_state}
                yield f"data: {json.dumps(payload)}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'text': 'Action performed...'})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'text': text, 'cached': True})}\n\n"

            chat_turns.append({
                "user": req.message, "assistant": text, "tool_calls": events
            })
            return

        # ── Cache MISS: stream from model ─────────────────────────────────────
        print(f"[Cache] MISS (stream) for: {resolved_message!r}")
        messages_out: list[dict] = []
        tool_events:  list[dict] = []
        initial_len = len(conversation_history)

        for event in run_agent_stream(
            resolved_message,
            history=conversation_history,
            backend=active_backend,
            messages_out=messages_out,
        ):
            if event["type"] == "tool_call":
                tool_events.append({
                    "name":   event["name"],
                    "args":   event["args"],
                    "result": event["result"],
                })
                # Add current home state to the event before streaming to frontend
                event["state"] = home_state
            yield f"data: {json.dumps(event)}\n\n"

            # After done, handle history + cache bookkeeping
            if event["type"] == "done":
                final_text = event["text"]

                # Sliding window — keep last 2 turns (4 messages)
                conversation_history.extend(messages_out[1 + initial_len:])
                if len(conversation_history) > 4:
                    conversation_history[:] = conversation_history[-4:]

                chat_turns.append({
                    "user":       req.message,
                    "assistant":  final_text,
                    "tool_calls": tool_events,
                })

                action_events = [
                    e for e in tool_events
                    if e["name"] != "intent_unclear"
                    and e.get("result", {}).get("success")
                ]
                if action_events:
                    to_cache = [{"name": e["name"], "args": e["args"]} for e in action_events]
                    set_cached(resolved_message, to_cache, current_snapshot)
                    print(f"[Cache] STORED {len(to_cache)} call(s) for: {resolved_message!r}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering if behind a proxy
        },
    )
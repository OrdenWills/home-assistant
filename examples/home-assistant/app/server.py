#app/server.py
import subprocess
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from app.agent import run_agent_stream, get_system_prompt
from app.state import (
    home_state, load_state,
    build_state_summary, log_action, build_action_log_context, action_log,
)
from app.cache import init_db, get_cached, set_cached, delete_cached, clear_all, list_entries, build_snapshot
from app.tools.handlers import TOOL_HANDLERS

# ── Dataset directory ──────────────────────────────────────────────────────────

DATASETS_DIR = "datasets"
GOLDEN_SET_PATH  = os.path.join(DATASETS_DIR, "golden_set.jsonl")
FAILURE_SET_PATH = os.path.join(DATASETS_DIR, "failure_set.jsonl")

os.makedirs(DATASETS_DIR, exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────


LOCAL_MODELS = [
    {"id":"home-assistant-sft","name":"Home Assistant SFT (Finetuned)","hf_repo":"OrdenWills/LFM2.5-1.2B-home-assistant-sft","hf_file":"LFM2.5-1.2B-Instruct.Q4_K_M.gguf","size_label":"714 MB","score_label":"99%"},
    {"id":"home-assistant-sft(small)","name":"Home(small) Assistant SFT (Finetuned)","hf_repo":"OrdenWills/LFM2.5-350M-home-assistant-sft","hf_file":"LFM2.5-350M.Q4_K_M.gguf","size_label":"218 MB","score_label":"98%"},
    {"id": "home-assistant-sft-small-8bit","name": "Home Assistant SFT (Finetuned 350M)","hf_repo": "OrdenWills/LFM2.5-350M-home-assistant-sft","hf_file": "LFM2.5-350M.Q8_0.gguf","size_label":"372 MB","score_label":"99%"},
    {"id":"lfm25-1b-thinking-q4","name":"LFM2.5-1.2B-Thinking-Q4_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Thinking-GGUF","hf_file":"LFM2.5-1.2B-Thinking-Q4_0.gguf","size_label":"718 MB","score_label":"75%"},
    {"id":"lfm25-1b-thinking-q8","name":"LFM2.5-1.2B-Thinking-Q8_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Thinking-GGUF","hf_file":"LFM2.5-1.2B-Thinking-Q8_0.gguf","size_label":"1.28 GB","score_label":"82%"},
    {"id":"lfm25-1b-q4","name":"LFM2.5-1.2B-Instruct-Q4_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Instruct-GGUF","hf_file":"LFM2.5-1.2B-Instruct-Q4_0.gguf","size_label":"696 MB","score_label":"68%"},
    {"id":"lfm25-1b-q8","name":"LFM2.5-1.2B-Instruct-Q8_0.gguf","hf_repo":"LiquidAI/LFM2.5-1.2B-Instruct-GGUF","hf_file":"LFM2.5-1.2B-Instruct-Q8_0.gguf","size_label":"1.25 GB","score_label":"53%"},
    {"id":"lfm2-vl-450m-q4","name":"LFM2-VL-450M-Q4_0.gguf","hf_repo":"LiquidAI/LFM2-VL-450M-GGUF","hf_file":"LFM2-VL-450M-Q4_0.gguf","size_label":"209 MB","score_label":"40%"},
]

# ── Module-level state ─────────────────────────────────────────────────────────

# Stores complete turns for history replay AND dataset export.
# Each entry: {turn_id, user, resolved_message, current_room, assistant, tool_calls}
chat_turns: list[dict] = []
active_backend: str = "local"

llama_proc: subprocess.Popen | None = None
llama_status: str = "idle"
llama_active_model_id: str | None = None
llama_error: str | None = None


# ── Relative-keyword detector (for short-circuit path) ─────────────────────────

_RELATIVE_KEYWORDS: set[str] = {
    "other", "the other", "every other", "all other",
    "the rest", "rest of", "remaining", "except", "apart from",
    "others", "the others",
}

def _has_relative_keyword(message: str) -> bool:
    lower = message.lower()
    return any(k in lower for k in _RELATIVE_KEYWORDS)


def _build_resolved_message(raw_message: str, current_room: str | None = None) -> str:
    """
    Build the full prompt sent to the agent by prefixing the user's raw message
    with the structured state summary and recent action log.
    """
    print(f"\n[AI Hit] Current Room: {current_room or 'None'}")
    state_str = build_state_summary(current_room)
    action_str = build_action_log_context()
    parts = [state_str]
    if action_str:
        parts.append(action_str)
    parts.append(raw_message)
    return "\n".join(parts)


def _new_turn_id() -> int:
    """Return the next sequential turn ID (= current length before append)."""
    return len(chat_turns)


# ── Dataset writer ─────────────────────────────────────────────────────────────

def _write_dataset_entry(turn: dict, rating: str) -> str:
    """
    Write a full training/debug entry to the appropriate JSONL file.
    Returns the path it was written to.
    """
    entry = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "turn_id":          turn["turn_id"],
        "rating":           rating,                         # "positive" | "negative"
        "backend":          active_backend,
        "model":            llama_active_model_id or "openai/gpt-4o-mini",
        # ── Everything the model actually saw ──
        "system_prompt":    turn.get("system_prompt", ""),
        "current_room":     turn.get("current_room"),
        "raw_user_message": turn["user"],
        "resolved_message": turn.get("resolved_message", ""),
        # ── What the model did ──
        "tool_calls":       turn.get("tool_calls", []),
        "final_response":   turn["assistant"],
    }

    dest = GOLDEN_SET_PATH if rating == "positive" else FAILURE_SET_PATH
    with open(dest, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Dataset] Written {rating} entry (turn {turn['turn_id']}) → {dest}")
    return dest


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
    action_log.clear()
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
    action_log.clear()
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

# ── Feedback / Dataset endpoint ────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    turn_id: int
    rating: str   # "positive" | "negative"

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Store the full turn context to golden_set.jsonl (👍) or failure_set.jsonl (👎).
    The stored entry includes the system prompt, resolved message (with [STATE:]
    and [RECENT ACTIONS:] prefixes), tool calls that were executed, and the final
    model response — i.e. everything the model saw and did.
    """
    if req.rating not in ("positive", "negative"):
        return JSONResponse({"error": "rating must be 'positive' or 'negative'"}, status_code=400)

    # Find the turn by turn_id (stored on every entry)
    turn = next((t for t in chat_turns if t.get("turn_id") == req.turn_id), None)
    if turn is None:
        return JSONResponse({"error": f"turn_id {req.turn_id} not found"}, status_code=404)

    dest = _write_dataset_entry(turn, req.rating)
    return JSONResponse({"ok": True, "stored_in": dest})

@app.get("/dataset/stats")
def dataset_stats():
    """Quick stats on how many entries are in each dataset file."""
    def count_lines(path: str) -> int:
        try:
            with open(path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0

    return JSONResponse({
        "golden_set":  {"path": GOLDEN_SET_PATH,  "entries": count_lines(GOLDEN_SET_PATH)},
        "failure_set": {"path": FAILURE_SET_PATH, "entries": count_lines(FAILURE_SET_PATH)},
    })

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
    if name == "control_tv":       return f"{args['room'].replace('_',' ').title()} TV turned {args['state']}."
    if name == "control_speaker":  return f"{args['room'].replace('_',' ').title()} speaker: {args['action']}."
    if name == "control_fan":      return f"{args['room'].replace('_',' ').title()} fan turned {args['state']}."
    if name == "intent_unclear":   return f"Intent unclear ({args.get('reason','?')})."
    return "Done."

# ── Chat endpoint ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    current_room: str | None = None



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

    # ── Short-circuit for relative keywords ───────────────────────────────────
    is_relative = _has_relative_keyword(req.message)
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

    # ── Build resolved message with state + action log ────────────────────────
    resolved_message = _build_resolved_message(req.message, req.current_room)

    # ── Cache check ───────────────────────────────────────────────────────────
    current_snapshot = build_snapshot(home_state)
    cached = get_cached(resolved_message, current_snapshot)

    avail_r = list(home_state["lights"].keys())
    avail_d = list(home_state["doors"].keys())
    tv_rooms = list(home_state.get("tv", {}).keys())
    spk_rooms = list(home_state.get("speaker", {}).keys())
    fan_rooms = list(home_state.get("fan", {}).keys())
    tv_str = ", ".join(tv_rooms) if tv_rooms else "none"
    spk_str = ", ".join(spk_rooms) if spk_rooms else "none"
    fan_str = ", ".join(fan_rooms) if fan_rooms else "none"
    system_prompt = get_system_prompt(avail_r, avail_d, tv_str, spk_str, fan_str)

    def generate():
        # ── Short-circuit path for relative keywords ──────────────────────────
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
                    summary = f"{room.replace('_', ' ').title()} light turned {target}."
                    summaries.append(summary)
                    log_action("toggle_lights", args, summary)

                text = " ".join(summaries)
                turn_id = _new_turn_id()
                chat_turns.append({
                    "turn_id": turn_id,
                    "user": req.message,
                    "system_prompt": "",
                    "resolved_message": "",
                    "current_room": req.current_room,
                    "assistant": text,
                    "tool_calls": events,
                })

                to_cache = [{"name": e["name"], "args": e["args"]} for e in events]
                set_cached(req.message, to_cache, current_snapshot)

                yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'text': text, 'turn_id': turn_id, 'cached': False})}\n\n"
            else:
                text = f"The other lights are already {target}."
                turn_id = _new_turn_id()
                chat_turns.append({
                    "turn_id": turn_id,
                    "user": req.message,
                    "system_prompt": "",
                    "resolved_message": "",
                    "current_room": req.current_room,
                    "assistant": text,
                    "tool_calls": [],
                })
                yield f"data: {json.dumps({'type': 'done', 'text': text, 'turn_id': turn_id, 'cached': False})}\n\n"
            return

        # ── Cache HIT: replay tools then emit done immediately ────────────────
        if cached:
            print(f"[Cache] HIT (stream) for: {req.message!r}")
            events, text = _replay_cached_tools(cached)
            for e in events:
                yield f"data: {json.dumps({'type': 'status', 'text': 'Performing action...'})}\n\n"
                payload = {"type": "tool_call", "name": e["name"],
                           "args": e["args"], "result": e["result"], "state": home_state}
                yield f"data: {json.dumps(payload)}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'text': 'Action performed...'})}\n\n"
                log_action(e["name"], e["args"], _summarise_tool(e["name"], e["args"], e["result"]))

            turn_id = _new_turn_id()
            chat_turns.append({
                "turn_id": turn_id,
                "user": req.message,
                "system_prompt": system_prompt,
                "resolved_message": resolved_message,
                "current_room": req.current_room,
                "assistant": text,
                "tool_calls": events,
            })

            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'text': text, 'turn_id': turn_id, 'cached': True})}\n\n"
            return

        # ── Cache MISS: stream from model ─────────────────────────────────────
        # The LLM call blocks 2-12 s.  We run the agent in a background
        # thread so the SSE generator can keep emitting rotating status
        # messages while we wait, keeping the UI feeling alive.
        import queue as _q

        print(f"[Cache] MISS (stream) for: {req.message!r}")

        event_q: _q.Queue = _q.Queue()

        def _agent_worker():
            try:
                for ev in run_agent_stream(
                    user_message=resolved_message,
                    system_prompt=system_prompt,
                    backend=active_backend
                ):
                    event_q.put(ev)
            except Exception as exc:
                event_q.put({"type": "error", "text": str(exc)})
            finally:
                event_q.put(None)   # sentinel

        threading.Thread(target=_agent_worker, daemon=True).start()

        _status_msgs = [
            "Understanding intent...",
            f"user said \"{req.message}\"",
            "Processing request...",
            "Analyzing command...",
            "Deciding next action...",
            "Generating response...",
            ""
        ]
        _si = 0                       # status index
        tool_events: list[dict] = []
        done_seen = False

        # First status fires instantly
        yield f"data: {json.dumps({'type': 'status', 'text': _status_msgs[0]})}\n\n"
        _si = 1

        while not done_seen:
            try:
                event = event_q.get(timeout=1.5)
            except _q.Empty:
                # Model still thinking — rotate to next status message
                yield f"data: {json.dumps({'type': 'status', 'text': _status_msgs[_si % len(_status_msgs)]})}\n\n"
                _si += 1
                continue

            if event is None:
                break                 # agent thread finished

            if event["type"] == "tool_call":
                yield f"data: {json.dumps({'type': 'status', 'text': 'Performing action...'})}\n\n"
                tool_events.append({
                    "name":   event["name"],
                    "args":   event["args"],
                    "result": event["result"],
                })
                event["state"] = home_state
                log_action(
                    event["name"], event["args"],
                    _summarise_tool(event["name"], event["args"], event["result"]),
                )
                yield f"data: {json.dumps(event)}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'text': 'Action performed ✓'})}\n\n"

            elif event["type"] == "done":
                done_seen = True
                final_text = event["text"]
                yield f"data: {json.dumps(event)}\n\n"

                turn_id = _new_turn_id()
                chat_turns.append({
                    "turn_id": turn_id,
                    "user": req.message,
                    "system_prompt": system_prompt,
                    "resolved_message": resolved_message,
                    "current_room": req.current_room,
                    "assistant": final_text,
                    "tool_calls": tool_events,
                })
                yield f"data: {json.dumps({'type': 'turn_id', 'turn_id': turn_id})}\n\n"

                action_events = [
                    e for e in tool_events
                    if e["name"] != "intent_unclear"
                    and e.get("result", {}).get("success")
                ]
                if action_events:
                    to_cache = [{"name": e["name"], "args": e["args"]} for e in action_events]
                    set_cached(resolved_message, to_cache, current_snapshot)
                    print(f"[Cache] STORED {len(to_cache)} call(s) for: {req.message!r}")

            else:
                # token, error — forward as-is
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
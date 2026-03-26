#app/server.py
import subprocess
import threading
import time
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from app.agent import local_client, run_agent
from app.state import home_state, load_state
from app.cache import (
    init_db, get_cached, set_cached, delete_cached,
    clear_all, clear_stale, list_entries, build_snapshot,
)
from app.tools.handlers import TOOL_HANDLERS

# ── Model registry ─────────────────────────────────────────────────────────────

LOCAL_MODELS = [
    {
        "id": "lfm25-1b-thinking-q4",
        "name": "LFM2.5-1.2B-Thinking-Q4_0.gguf",
        "hf_repo": "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        "hf_file": "LFM2.5-1.2B-Thinking-Q4_0.gguf",
        "size_label": "718 MB",
        "score_label": "75%",
    },
    {
        "id": "lfm25-1b-thinking-q8",
        "name": "LFM2.5-1.2B-Thinking-Q8_0.gguf",
        "hf_repo": "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        "hf_file": "LFM2.5-1.2B-Thinking-Q8_0.gguf",
        "size_label": "1.28 GB",
        "score_label": "82%",
    },
    {
        "id": "lfm25-1b-q4",
        "name": "LFM2.5-1.2B-Instruct-Q4_0.gguf",
        "hf_repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "hf_file": "LFM2.5-1.2B-Instruct-Q4_0.gguf",
        "size_label": "696 MB",
        "score_label": "68%",
    },
    {
        "id": "lfm25-1b-q8",
        "name": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
        "hf_repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "hf_file": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
        "size_label": "1.25 GB",
        "score_label": "53%",
    },
    {
        "id": "lfm2-vl-450m-q4",
        "name": "LFM2-VL-450M-Q4_0.gguf",
        "hf_repo": "LiquidAI/LFM2-VL-450M-GGUF",
        "hf_file": "LFM2-VL-450M-Q4_0.gguf",
        "size_label": "209 MB",
        "score_label": "40%",
    },
]

# ── Module-level state ─────────────────────────────────────────────────────────

conversation_history: list[dict] = []
active_backend: str = "local"

llama_proc: subprocess.Popen | None = None
llama_status: str = "idle"          # idle | starting | ready | error
llama_active_model_id: str | None = None
llama_error: str | None = None


# ── Background thread helper ───────────────────────────────────────────────────

def _start_llama_server_bg(model: dict) -> None:
    global llama_proc, llama_status, llama_active_model_id, llama_error, active_backend

    llama_status = "starting"
    llama_active_model_id = model["id"]
    llama_error = None

    if llama_proc is not None:
        try:
            llama_proc.terminate()
            llama_proc.wait(timeout=10)
        except Exception:
            llama_proc.kill()
        llama_proc = None

    print(f"[LlamaServer] Starting: {model['name']}")

    model_path = None
    try:
        model_path = hf_hub_download(
            repo_id=model["hf_repo"],
            filename=model["hf_file"],
            repo_type="model",
            local_files_only=True,
        )
        print(f"[LlamaServer] Using cached model: {model_path}")
    except Exception:
        print(f"[LlamaServer] Model not in cache, will download on demand...")
        model_path = None

    if model_path:
        cmd = [
            "C:\\Users\\Adolphus\\llama-b8479-bin-win-cpu-x64\\llama-server.exe",
            "--model", model_path,
            "--port", "8080",
            "--ctx-size", "4096",
            "--n-gpu-layers", "99",
        ]
    else:
        cmd = [
            "C:\\Users\\Adolphus\\llama-b8479-bin-win-cpu-x64\\llama-server.exe",
            "--hf-repo", model["hf_repo"],
            "--hf-file", model["hf_file"],
            "--port", "8080",
            "--ctx-size", "4096",
            "--n-gpu-layers", "99",
        ]

    print(f"[LlamaServer] Command: {' '.join(cmd)}")

    try:
        llama_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def log_output():
            try:
                for line in iter(llama_proc.stdout.readline, ""):
                    if line:
                        print(f"[LlamaServer] {line.rstrip()}")
            except Exception:
                pass

        import threading as th
        th.Thread(target=log_output, daemon=True).start()

    except Exception as e:
        llama_status = "error"
        llama_error = str(e)
        print(f"[LlamaServer] Error starting process: {e}")
        return

    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8080/v1/models", timeout=2)
            llama_status = "ready"
            active_backend = "local"
            elapsed = time.time() - (deadline - 1800)
            print(f"[LlamaServer] Ready! Loaded in {elapsed:.1f}s")
            return
        except Exception:
            time.sleep(2)

    llama_status = "error"
    llama_error = "llama-server did not become ready within 1800s"
    print(f"[LlamaServer] {llama_error}")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Initialise the SQLite cache and restore persisted home state
    init_db()
    print("[Cache] SQLite tool-call cache initialised.")
    load_state()

    # Auto-load default model on startup
    default_model_id = "lfm25-1b-q8"
    model = next((m for m in LOCAL_MODELS if m["id"] == default_model_id), None)
    if model:
        print(f"[Startup] Auto-loading default model: {model['name']}")
        threading.Thread(
            target=_start_llama_server_bg, args=(model,), daemon=True
        ).start()

    yield

    global llama_proc
    if llama_proc is not None:
        try:
            llama_proc.terminate()
            llama_proc.wait(timeout=10)
        except Exception:
            llama_proc.kill()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ── Existing endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/model")
def get_model():
    global active_backend, llama_active_model_id
    if active_backend == "openai":
        return JSONResponse({"name": "gpt-4o-mini"})
    if llama_active_model_id:
        model = next(
            (m for m in LOCAL_MODELS if m["id"] == llama_active_model_id), None
        )
        if model:
            return JSONResponse({"name": model["name"]})
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


@app.post("/reset")
def reset():
    conversation_history.clear()
    return JSONResponse({"ok": True})


# ── Local model endpoints ──────────────────────────────────────────────────────

@app.get("/local-models")
def get_local_models():
    return JSONResponse(LOCAL_MODELS)


@app.get("/local-model-status")
def get_local_model_status():
    return JSONResponse({
        "status":   llama_status,
        "model_id": llama_active_model_id,
        "error":    llama_error,
    })


class LocalModelRequest(BaseModel):
    model_id: str


@app.post("/local-model")
def start_local_model(req: LocalModelRequest):
    model = next((m for m in LOCAL_MODELS if m["id"] == req.model_id), None)
    if model is None:
        return JSONResponse({"error": "unknown model_id"}, status_code=400)
    threading.Thread(
        target=_start_llama_server_bg, args=(model,), daemon=True
    ).start()
    return JSONResponse({"status": "starting"})


@app.delete("/local-model")
def stop_local_model():
    global llama_proc, llama_status, llama_active_model_id, active_backend
    if llama_proc is not None:
        try:
            llama_proc.terminate()
            llama_proc.wait(timeout=10)
        except Exception:
            llama_proc.kill()
        llama_proc = None
    llama_status = "idle"
    llama_active_model_id = None
    active_backend = "openai"
    return JSONResponse({"status": "idle"})


# ── Cache endpoints ────────────────────────────────────────────────────────────

@app.get("/cache")
def list_cache():
    """Return every entry stored in the tool-call cache, with is_stale flags."""
    snapshot = build_snapshot(home_state)
    return JSONResponse(list_entries(current_snapshot=snapshot))


@app.delete("/cache")
def wipe_cache():
    """Wipe the entire tool-call cache."""
    deleted = clear_all()
    return JSONResponse({"deleted": deleted})


class CacheDeleteRequest(BaseModel):
    message: str


@app.delete("/cache/entry")
def delete_cache_entry(req: CacheDeleteRequest):
    """Remove a single cache entry by its (unnormalised) message text."""
    removed = delete_cached(req.message)
    return JSONResponse({"removed": removed})


@app.get("/cache/stale")
def list_stale_cache():
    """
    Return every cache entry that is stale relative to the current device
    topology (i.e. a room or door has been added / removed since it was cached).
    """
    snapshot = build_snapshot(home_state)
    entries  = list_entries(current_snapshot=snapshot)
    stale    = [e for e in entries if e.get("is_stale")]
    return JSONResponse({"count": len(stale), "entries": stale})


@app.delete("/cache/stale")
def clean_stale_cache():
    """
    Delete all stale cache entries.  Safe to call any time; the next request
    for a stale phrase will go to the model and rebuild a correct entry.
    """
    snapshot = build_snapshot(home_state)
    deleted  = clear_stale(snapshot)
    return JSONResponse({"deleted": deleted})


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _replay_cached_tools(
    cached_calls: list[dict],
    on_tool_call=None,
) -> tuple[list[dict], str]:
    """
    Execute a list of cached {name, args} tool calls against the live handlers.
    Returns (events_list, human_readable_summary).
    State is mutated exactly as it would be during a live agent turn.
    """
    events: list[dict] = []
    summaries: list[str] = []

    for tc in cached_calls:
        name = tc["name"]
        args = tc.get("args", {})
        handler = TOOL_HANDLERS.get(name)
        result = handler(**args) if handler else {"error": f"Unknown tool: {name}"}

        events.append({"name": name, "args": args, "result": result})
        if on_tool_call:
            on_tool_call(name, args, result)

        summaries.append(_summarise_tool(name, args, result))

    text = " ".join(summaries) if summaries else "Done."
    return events, text


def _summarise_tool(name: str, args: dict, result: dict) -> str:
    """Build a short human-readable sentence for a single tool call result."""
    if not result.get("success"):
        return f"({name} failed)"

    if name == "toggle_lights":
        return f"{args['room'].replace('_', ' ').title()} light turned {args['state']}."
    if name == "toggle_all_lights":
        return f"All lights turned {args['state']}."
    if name == "lock_door":
        action = "locked" if args["state"] == "lock" else "unlocked"
        return f"{args['door'].title()} door {action}."
    if name == "lock_all_doors":
        action = "locked" if args["state"] == "lock" else "unlocked"
        return f"All doors {action}."
    if name == "set_thermostat":
        return (
            f"Thermostat set to {args['temperature']}°F "
            f"in {args['mode']} mode."
        )
    if name == "set_scene":
        return f"{args['scene'].replace('_', ' ').title()} scene activated."
    if name == "intent_unclear":
        return f"Intent unclear ({args.get('reason', '?')})."
    return "Done."


# ── Chat endpoint ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    # ── 1. Gate: local model must be ready ────────────────────────────────────
    if active_backend == "local" and llama_status != "ready":
        msg = {
            "starting": "Model is still loading, please wait.",
            "idle":     "No local model loaded. Select a model from the LFM Local dropdown.",
            "error":    f"Local model failed to start: {llama_error}",
        }.get(llama_status, "Local model is not ready.")
        return JSONResponse({"text": msg, "tool_calls": []}, status_code=503)

    # ── 2. Cache lookup ───────────────────────────────────────────────────────
    current_snapshot = build_snapshot(home_state)
    cached = get_cached(req.message, current_snapshot)
    if cached:
        print(f"[Cache] HIT for: {req.message!r}")
        events, text = _replay_cached_tools(cached)
        return JSONResponse({
            "text":       text,
            "tool_calls": events,
            "cached":     True,         # lets the frontend know (optional)
        })

    # ── 3. Cache miss — run the agent normally ────────────────────────────────
    print(f"[Cache] MISS for: {req.message!r} — invoking model")
    events: list[dict] = []

    def on_tool_call(name, args, result):
        print(f"[Server] Tool called: {name} with args {args}, result: {result}")
        events.append({"name": name, "args": args, "result": result})

    initial_history_len = len(conversation_history)
    messages_out: list[dict] = []
    try:
        text = run_agent(
            req.message,
            history=conversation_history,
            backend=active_backend,
            on_tool_call=on_tool_call,
            messages_out=messages_out,
        )
    except Exception as e:
        return JSONResponse({"text": f"Error: {e}", "tool_calls": events}, status_code=500)

    # Extend conversation history with only the new messages from this turn
    new_messages_start = 1 + initial_history_len
    conversation_history.extend(messages_out[new_messages_start:])

    # ── 4. Persist to cache only when the model called at least one tool ──────
    # intent_unclear counts as a tool call but we don't want to cache it,
    # because the user's next attempt may carry enough context to resolve it.
    action_events = [
        e for e in events
        if e["name"] != "intent_unclear"
        and e.get("result", {}).get("success")
    ]
    if action_events:
        to_cache = [{"name": e["name"], "args": e["args"]} for e in action_events]
        set_cached(req.message, to_cache, current_snapshot)
        print(f"[Cache] STORED {len(to_cache)} tool call(s) for: {req.message!r}")

    return JSONResponse({"text": text, "tool_calls": events, "cached": False})
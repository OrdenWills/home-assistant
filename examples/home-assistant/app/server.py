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
from app.state import home_state

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
    "score_label": "40%",},
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
    
    # Try to resolve cached model first (for pre-downloaded models)
    model_path = None
    try:
        model_path = hf_hub_download(
            repo_id=model["hf_repo"],
            filename=model["hf_file"],
            repo_type="model",
            local_files_only=True,  # Don't download, only check cache
        )
        print(f"[LlamaServer] Using cached model: {model_path}")
    except Exception:
        # Not in cache, will use repo/file method (llama-server downloads on first run)
        print(f"[LlamaServer] Model not in cache, will download on demand...")
        model_path = None

    # Build command
    if model_path:
        # Use cached file directly
        cmd = [
            "C:\\Users\\Adolphus\\llama-b8479-bin-win-cpu-x64\\llama-server.exe",
            "--model", model_path,
            "--port", "8080",
            "--ctx-size", "4096",
            "--n-gpu-layers", "99",
        ]
    else:
        # Fall back to repo/file (llama-server will download)
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
        # Capture stderr to see what's happening
        llama_proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time for debugging
        def log_output():
            try:
                for line in iter(llama_proc.stdout.readline, ''):
                    if line:
                        print(f"[LlamaServer] {line.rstrip()}")
            except Exception:
                pass
        
        import threading as th
        log_thread = th.Thread(target=log_output, daemon=True)
        log_thread.start()
        
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
    # Auto-load default model on startup
    default_model_id = "lfm25-1b-q8"
    model = next((m for m in LOCAL_MODELS if m["id"] == default_model_id), None)
    if model:
        print(f"[Startup] Auto-loading default model: {model['name']}")
        threading.Thread(target=_start_llama_server_bg, args=(model,), daemon=True).start()
    
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
    # Return the actual selected model name from the registry
    if llama_active_model_id:
        model = next((m for m in LOCAL_MODELS if m["id"] == llama_active_model_id), None)
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
        "status": llama_status,
        "model_id": llama_active_model_id,
        "error": llama_error,
    })


class LocalModelRequest(BaseModel):
    model_id: str


@app.post("/local-model")
def start_local_model(req: LocalModelRequest):
    model = next((m for m in LOCAL_MODELS if m["id"] == req.model_id), None)
    if model is None:
        return JSONResponse({"error": "unknown model_id"}, status_code=400)
    threading.Thread(target=_start_llama_server_bg, args=(model,), daemon=True).start()
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


# ── Chat endpoint ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    if active_backend == "local" and llama_status != "ready":
        msg = {
            "starting": "Model is still loading, please wait.",
            "idle": "No local model loaded. Select a model from the LFM Local dropdown.",
            "error": f"Local model failed to start: {llama_error}",
        }.get(llama_status, "Local model is not ready.")
        return JSONResponse({"text": msg, "tool_calls": []}, status_code=503)

    events = []

    def on_tool_call(name, args, result):
        print(f"[Server] Tool called: {name} with args {args}, result: {result}")
        events.append({"name": name, "args": args, "result": result})

    messages_out = []
    try:
        text = run_agent(req.message, history=conversation_history, backend=active_backend, on_tool_call=on_tool_call, messages_out=messages_out)
    except Exception as e:
        return JSONResponse({"text": f"Error: {e}", "tool_calls": events}, status_code=500)

    # Append the full conversation (including tool calls) to history
    # This ensures the model knows about actual tool calls in future requests
    conversation_history.extend(messages_out)

    return JSONResponse({"text": text, "tool_calls": events})

# Home Assistant Local LFM Application

A fully local smart home assistant powered by a purpose-trained LFM2.5 model. This application demonstrates a production-ready home automation system that runs entirely on your own hardware, with no dependency on cloud APIs.

> **Note:** The original cookbook and tutorial content (benchmarking, synthetic data generation, fine-tuning steps) remains in the root directory. This README documents only the current application state: `app/`, `index.html`, `style.css`, and `datasets/`.

---

## 🚀 Quick Start

**Requirements:**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- [llama.cpp](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#installation) for local model inference (`llama-server` on PATH)

**Start the application:**

```bash
# Activate environment and start FastAPI server
uv run uvicorn app.server:app --port 5173 --reload
```

**Open in browser:**

```
http://localhost:5173
```

The UI features an integrated model selector. Choose a model, and the app automatically downloads and launches `llama-server` in the background.

**Pre-download models (recommended):**

To avoid download timeouts on first model selection, pre-cache models:

```bash
# PowerShell (Windows)
.\download_models.ps1

# Batch (Windows)
download_models.bat

# Python (Any OS)
uv run python download_models.py
```

First-time downloads may take 30–60 minutes depending on connection speed.

---

## 📐 Architecture

The application is built around a clean separation of concerns:

```
┌──────────────────────────────────────────────────────────────┐
│                    Browser UI (HTML/CSS/JS)                  │
│          Renders device state and sends chat messages        │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTP / Server-Sent Events
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Server (8000)                      │
│  ─────────────────────────────────────────────────────────  │
│  ├─ Endpoint: POST /chat (streaming chat requests)          │
│  ├─ Endpoint: GET /state (pull current home state)          │
│  ├─ Endpoint: GET /history (chat history retrieval)         │
│  ├─ Endpoint: Datasets API (golden_set, failure_set export) │
│  └─ Model Registry (auto-downloads & launches llama.cpp)    │
└────────────────────┬─────────────────────────────────────────┘
                     │ (spawns background process)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│          Agent Loop (Orchestrates conversation flow)         │
│  ─────────────────────────────────────────────────────────  │
│  ├─ System prompt construction (device topology)            │
│  ├─ Streaming inference with tool schemas                  │
│  ├─ Tool call extraction & validation                       │
│  ├─ Recursive tool execution (handles multi-turn)           │
│  └─ Response formatting & persistence                       │
└────────────────────┬─────────────────────────────────────────┘
             ┌───────┴────────┬────────────────┐
             │                │                │
             ▼                ▼                ▼
    ┌───────────────┐ ┌─────────────┐ ┌──────────────┐
    │  Home State   │ │ Tool Schemas│ │Tool Handlers │
    │   (SQLite)    │ │  & Coercion │ │ & Validators │
    │               │ │             │ │              │
    │ • lights      │ │ • JSON spec │ │ • Mutators   │
    │ • doors       │ │ • 10 tools  │ │ • Verifiers  │
    │ • thermostat  │ │ • Params    │ │ • Exceptions │
    │ • scenes      │ │             │ │              │
    │ • TV/speaker  │ └─────────────┘ └──────────────┘
    │ • fans        │
    └───────────────┘
             │
             └─ Local SQLite DB (cache.db)
```

### Key Components

#### 1. **FastAPI Server** (`app/server.py`)
- Hosts HTTP endpoints for chat, state retrieval, and dataset export
- Manages model lifecycle (download, launch, termination)
- Implements OpenAI-compatible API bridge to `llama.cpp`
- Integrates with SQLite for state and cache persistence

**Model Registry:**
The app includes 8 pre-configured models:

| Model | Size | Quantization | Score |
|-------|------|-------------|-------|
| Home Assistant SFT (1.2B) | 714 MB | Q4_K_M | **95%** |
| Home Assistant SFT (350M) | 218 MB | Q4_K_M | **96%** |
| Home Assistant SFT (350M) | 372 MB | Q8_0 | **96%** |
| LFM2.5-1.2B-Thinking | 718 MB | Q4_0 | 75% |
| LFM2.5-1.2B-Thinking | 1.28 GB | Q8_0 | 82% |
| LFM2.5-1.2B-Instruct | 696 MB | Q4_0 | 68% |
| LFM2.5-1.2B-Instruct | 1.25 GB | Q8_0 | 53% |
| LFM2-VL-450M | 209 MB | Q4_0 | 40% |

**Recommended models for production:** The 350M variants (`home-assistant-sft`) achieve >96% accuracy and run efficiently on consumer hardware.

#### 2. **Agent Loop** (`app/agent.py`)
Implements a stateful conversation agent with:
- **System prompt synthesis** — Dynamically builds context from connected devices, scene options, and resolution rules
- **Streaming inference** — Supports OpenAI-compatible APIs (local `llama.cpp` and OpenAI as fallback)
- **Tool extraction & validation** — Parses JSON tool calls, coerces types, validates schemas
- **Recursive execution** — Handles multi-step actions (e.g., "lock the front door and turn off the lights")
- **State grounding** — Always provides `[STATE: ...]` prefix so the model knows current device conditions

The agent supports **two backends:**
- `local`: llama.cpp server (default)
- `openai`: GPT-4o-mini (for comparison/debugging)

#### 3. **Home State** (`app/state.py`)
Single source of truth for device conditions, persisted to SQLite:

```python
{
  "lights": {
    "living_room": {"state": "on|off"},
    "bedroom": {"state": "on|off"},
    # ... 6 rooms total
  },
  "doors": {
    "front": "locked|unlocked",
    "back": "locked|unlocked",
    # ... 9 doors total (room-specific locks)
  },
  "thermostat": {
    "temperature": 60–80 (°F),
    "mode": "heat|cool|auto"
  },
  "active_scene": "movie_night|bedtime|morning|away|party|None",
  "tv": {
    "living_room": "on|off",
  },
  "speaker": {
    "living_room": "playing|paused|stopped",
  },
  "fan": {
    "room": {"state": "on|off", "speed": "low|medium|high"},
  },
  "music_folder": Path | None,
  "current_track_index": int,
}
```

State mutations are atomic and immediately persisted to the database, ensuring consistency across server restarts.

#### 4. **Tool System** (`app/tools/`)
The agent can invoke 8 tools:

| Tool | Parameters | Notes |
|------|-----------|-------|
| `toggle_lights` | room, state | On/off control for 6 connected rooms |
| `lock_door` | door, state | Lock/unlock 9 doors (including room-specific) |
| `set_thermostat` | temperature (60–80°F), mode | Heat/cool/auto modes |
| `set_scene` | scene | Activates presets: movie_night, bedtime, morning, away, party |
| `control_tv` | room, state | On/off for living_room, bedroom |
| `control_fan` | room, state, speed | Speed control: low, medium, high |
| `control_speaker` | room, action, media | Play/pause/stop/next/previous + optional media parameter |
| `intent_unclear` | reason | Rejection tool: off_topic, incomplete, unsupported_device, unsupported_feature |

Each tool has:
- **Schema** (`app/tools/schemas.py`) — JSON specification for the model
- **Handler** (`app/tools/handlers.py`) — Implementation + state mutation
- **Validator** (`app/tools/validator.py`) — Type coercion and runtime checks

#### 5. **Cache Layer** (`app/cache.py`)
SQLite-backed caching system for:
- Model metadata snapshots
- Chat history per session
- Inference results (for testing/debugging)

#### 6. **Event System** (`app/events.py`)
Real-time event streaming to the frontend:
- Device state changes
- Chat messages
- Model status
- Tool execution results

---

## 🎨 User Interface

### File Structure
- **`index.html`** — Single-page application shell; imports CSS and JavaScript
- **`style.css`** — Responsive design system for desktop/mobile

### Features

#### Layout
- **3-row × 2-column cockpit grid** (desktop)
  1. Floor plan (spans both columns)
  2. Device control panels (lights, doors, thermostat, scenes)
  3. Chat interface (left), device status readout (right)
  
- **Mobile overlay** — Floating action button reveals chat on small screens

#### Controls
- **Floor Plan** — Visual representation of rooms; click to set `current_user_room`
- **Light Toggles** — Per-room on/off switches
- **Door Locks** — Visual lock/unlock controls
- **Thermostat** — Temperature slider (60–80°F) + mode selector
- **Scene Buttons** — One-click presets (Movie Night, Bedtime, etc.)
- **Model Selector** — Dropdown to swap between 8 pre-configured models
- **Chat Interface** — Streaming message display + input field
- **Theme Toggle** — Light/Dark/High-Contrast modes

#### Responsiveness
The UI adapts to mobile screens:
- Primary controls remain visible
- Chat slides into an overlay with backdrop
- FAB (Floating Action Button) provides quick access

---

## 📊 Datasets

### Structure
Located in `datasets/`:

- **`golden_set.jsonl`** — Successful interactions (ground truth for model evaluation)
- **`failure_set.jsonl`** — Failed or edge-case interactions (for debugging)

Each line is a JSON record:

```json
{
  "message": "Turn off the bedroom light",
  "state": "[STATE: lights={bedroom:on, ...}, ...]",
  "expected_tool_call": "toggle_lights",
  "expected_args": {"room": "bedroom", "state": "off"},
  "actual_tool_call": "toggle_lights",
  "actual_args": {"room": "bedroom", "state": "off"},
  "success": true,
  "timestamp": "2026-06-01T12:34:56Z"
}
```

### Purpose
- **Golden Set** — Used to validate model behavior; should remain 100% accurate
- **Failure Set** — Logged when the model produces unexpected tool calls or rejects valid requests; used for model retraining or prompt refinement

---

## 🧠 Fine-Tuned Model

### LFM2.5-350M Home Assistant SFT

The application ships with two production-grade fine-tuned variants:

**Base Model:** [LiquidAI/LFM2.5-350M](https://huggingface.io/LiquidAI/LFM2.5-350M)

**Fine-Tuned Variants:**
1. **LFM2.5-350M-home-assistant-sft** (218 MB) — **96% accuracy** on task suite

**Training Dataset:** 157,000 synthetic examples across 33 instruction schemas

**Key Capabilities:**

| Capability | Example | Handled? |
|------------|---------|----------|
| Direct commands | "Turn on the bedroom light" | ✅ Yes |
| Already-satisfied detection | "Turn on the lights" when all are on | ✅ Yes |
| Pronoun resolution | "Turn it off" (current_user_room) | ✅ Yes |
| State-aware disambiguation | "Turn off the TV" (infers living_room) | ✅ Yes |
| Compound commands | "Lock doors and turn off lights" | ✅ Yes |
| Action log / undo | "Undo that" (reverses recent action) | ✅ Yes |
| Scene activation | "Movie night" | ✅ Yes |
| Rejection (off-topic) | "Order me pizza" | ✅ Rejects with `intent_unclear` |
| Rejection (ambiguous) | "Turn off the speaker" (multiple speakers) | ✅ Rejects with `intent_unclear` |

**System Prompt:**
The model is primed with a detailed system prompt that includes:
- Connected device topology
- Tool schemas and parameter constraints
- State-aware resolution rules for pronouns and implicit references
- Action log (recent transactions) for undo/repeat logic
- Synonym mappings (e.g., "open" = "unlock", "close" = "lock")

See [model card](./multi-step/model-card.md) for full training details.

---

## 🔄 Conversation Flow

1. **User sends message** (via chat input)
2. **Server receives POST /chat**
3. **Agent constructs system prompt**
   - Includes current `[STATE: ...]`
   - Includes `[RECENT ACTIONS: ...]` (for undo/repeat)
   - Includes list of connected devices
4. **Agent streams inference** from local llama.cpp
5. **Agent extracts tool calls** from streamed JSON
6. **Agent validates & coerces** tool arguments
7. **Agent executes tool** (mutates state)
8. **Agent re-prompts model** with tool result (if recursive action needed)
9. **Agent streams final text response** to browser
10. **Browser updates UI** with new state + message
11. **State persisted** to SQLite

---

## 🛠️ Development

### File Organization

```
app/
├── server.py           # FastAPI routes & model lifecycle
├── agent.py            # Conversation agent & streaming
├── state.py            # Home state schema & persistence
├── cache.py            # SQLite cache layer
├── events.py           # Real-time event system
└── tools/
    ├── __init__.py
    ├── schemas.py      # Tool JSON schemas
    ├── handlers.py     # Tool implementations
    └── validator.py    # Type coercion & validation

index.html             # Single-page app shell
style.css              # Responsive design system
datasets/
├── golden_set.jsonl   # Successful interactions
└── failure_set.jsonl  # Failed interactions
```

### Key Entry Points

- **Start server:** `uvicorn app.server:app --port 5173 --reload`
- **Run tests:** `uv run pytest tests/`
- **Check state:** `python -c "from app.state import load_state; load_state(); print(home_state)"`

### Adding a New Tool

1. Define schema in `app/tools/schemas.py` (JSON object + parameter spec)
2. Implement handler in `app/tools/handlers.py` (function that mutates state)
3. Update system prompt in `app/agent.py` to include the new tool
4. Add test cases to `datasets/golden_set.jsonl`
5. Restart server; model will see new tool in available actions

---

## 🐛 Debugging

### Check Home State
```bash
python -c "
from app.state import load_state
load_state()
from app.state import home_state
import json
print(json.dumps(home_state, indent=2))
"
```

### View Chat History
```bash
python -c "
from app.state import load_chat_history
history = load_chat_history()
for msg in history:
    print(msg)
"
```

### Check Cache
```bash
python -c "
from app.cache import list_entries
for entry in list_entries():
    print(entry)
"
```

### Model Inference Test
Switch to `openai` backend in `app/agent.py` to test against GPT-4o-mini:

```python
# In app/agent.py
# Change: run_agent_stream(..., backend='local')
# To:     run_agent_stream(..., backend='openai')
```

---

## 📝 Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI backend (optional, for GPT-4o-mini testing)
OPENAI_API_KEY=sk-...

# Model cache directory (optional)
MODEL_CACHE_DIR=./models

# Server port (optional, default 5173)
SERVER_PORT=5173
```

---

## ⚠️ Known Limitations

- **Temperature range fixed at 60–80°F** — Out-of-range requests produce a text explanation, not a tool call
- **No brightness/color control** — Dimming and color-change requests trigger `intent_unclear(unsupported_feature)`
- **English only** — All training data is English; other languages untested
- **Music playback limited to local library** — The `media` parameter for speaker control maps only to pre-indexed tracks
- **State must be accurate** — Stale `[STATE: ...]` causes incorrect disambiguation or rejection

---

## 🎯 What Changed from the Original README

The original README documented a **tutorial and benchmarking framework** for building a local home assistant. This APP_README documents the **completed application**:

| Aspect | Original | Current App |
|--------|----------|-------------|
| Purpose | Tutorial/cookbook | Production-ready application |
| Focus | Benchmarking & fine-tuning steps | Functional smart home control |
| Model | Multiple base models (1.2B, 350M) | Pre-fine-tuned home assistant models |
| Datasets | Generation scripts | Golden/failure sets for validation |
| UI | Mentioned as part of tutorial | Fully responsive web interface |
| State | Simple in-memory | Persistent SQLite database |
| Tools | 5 basic tools | 8 specialized tools + advanced resolution |
| Caching | Not mentioned | SQLite-backed cache layer |
| Events | Not mentioned | Real-time event streaming |

---

## 📚 References

- **Model Card:** [./multi-step/model-card.md](./multi-step/model-card.md)
- **LiquidAI LFM2.5:** [huggingface.co/LiquidAI](https://huggingface.co/LiquidAI)
- **llama.cpp:** [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **FastAPI:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

## 📜 License

Apache 2.0 (same as base model and dependencies)

---

**Last Updated:** June 1, 2026  
**Application Version:** 1.0 (Production Preview)

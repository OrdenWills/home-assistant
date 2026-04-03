# app/agent.py
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from app.tools.schemas import TOOL_SCHEMAS
from app.tools.handlers import TOOL_HANDLERS
from app.tools.validator import coerce_args, validate_tool_call  # Fix 1 & 2

load_dotenv()

local_client  = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

BACKENDS = {
    "local":  {"client": local_client,  "model": "local"},
    "openai": {"client": openai_client, "model": "gpt-4o-mini"},
}

# ── Fix 4: Tightened system prompt ────────────────────────────────────────────
# Key additions vs the old prompt:
#  - Explicit param names in every tool description
#  - A ❌ / ✓ bad-vs-good example for the exact failure seen in logs
#  - Reinforced that state= is always the on/off or lock/unlock parameter

SYSTEM_PROMPT = (
    "You are a home assistant AI. Use tools to control the home; respond in plain text when no tool is needed.\n"
    "You can output MULTIPLE tool calls in a single turn if needed.\n"
    "\n"
    "ROOMS: bedroom, bathroom, office, hallway, kitchen, living_room\n"
    "DOORS: front, back, garage, side\n"
    "\n"
    "TOOLS — use EXACTLY these parameter names:\n"
    "  toggle_lights(room=<room>, state='on'|'off')          — control a specific room\n"
    "  toggle_all_lights(state='on'|'off')                   — ALL rooms at once\n"
    "  lock_door(door=<door>, state='lock'|'unlock')         — control a specific door\n"
    "  lock_all_doors(state='lock'|'unlock')                 — ALL doors at once\n"
    "  set_thermostat(temperature=<60-80>, mode='heat'|'cool'|'auto')\n"
    "  set_scene(scene='movie_night'|'bedtime'|'morning'|'away'|'party')\n"
    "  intent_unclear(reason=<string>)\n"
    "\n"
    "CRITICAL — parameter names matter:\n"
    "  ❌ WRONG: [toggle_all_lights(room='off')]    ← 'room' is not a valid param\n"
    "  ✓ RIGHT:  [toggle_all_lights(state='off')]   ← always use 'state' for on/off\n"
    "  ❌ WRONG: [lock_all_doors(door='lock')]       ← 'door' is not a valid param here\n"
    "  ✓ RIGHT:  [lock_all_doors(state='lock')]      ← always use 'state' for lock/unlock\n"
    "\n"
    "ROUTING RULES:\n"
    "  - 'turn on/off all lights' or 'all lights on/off' → toggle_all_lights(state=...)\n"
    "  - 'turn on/off the light' with a room context     → toggle_lights(room=..., state=...)\n"
    "  - 'lock/unlock all doors'                         → lock_all_doors(state=...)\n"
    "  - 'lock/unlock the [door] door'                   → lock_door(door=..., state=...)\n"
    "\n"
    "Call intent_unclear (never plain text) when the request is: ambiguous, off-topic, "
    "incomplete (no target specified), or refers to an unsupported device (brightness, TV, music, etc.)."
)


def get_model_name(backend: str) -> str:
    if backend == "local":
        try:
            models = local_client.models.list()
            return models.data[0].id.split("_", 2)[-1] if models.data else "unknown"
        except Exception:
            return "unknown"
    return BACKENDS[backend]["model"]


def parse_tool_calls_from_text(text: str) -> list[dict]:
    """
    Parse tool calls from text.
    Handles both [func(args)] and [func(args), func(args)] formats safely.
    """
    tool_calls = []
    
    # Restrict parsing to known tools so we don't accidentally parse normal English words
    valid_tools = {
        "toggle_lights", "toggle_all_lights", "lock_door", 
        "lock_all_doors", "set_thermostat", "set_scene", "intent_unclear"
    }
    
    # Matches func_name(...) without getting confused by outer brackets.
    # [^)]* ensures we stop exactly at the closing parenthesis of EACH function.
    pattern = r'(\w+)\(([^)]*)\)'
    
    for idx, match in enumerate(re.finditer(pattern, text)):
        func_name = match.group(1)
        args_str  = match.group(2)
        
        if func_name not in valid_tools:
            continue
            
        try:
            args: dict = {}
            # Match double or single quoted strings: room="kitchen" or room='kitchen'
            for key, val in re.findall(r'(\w+)=["\']([^"\']*)["\']', args_str):
                args[key] = val
            # Match unquoted numbers: temperature=72
            for key, val in re.findall(r"(\w+)=([0-9]+(?:\.[0-9]+)?)\b", args_str):
                if key not in args:
                    args[key] = val
                    
            tool_calls.append({"name": func_name, "args": args, "id": f"call_{idx}"})
            print(f"[ParseToolCalls] Extracted: {func_name}({args})")
        except Exception as e:
            print(f"[ParseToolCalls] Failed to parse '{match.group(0)}': {e}")
            
    return tool_calls

def clean_text_response(text: str) -> str:
    """Remove tool call syntax and JSON objects from a text response."""
    # Remove tool call syntax: [function_name(...)]
    text = re.sub(r'\[(\w+)\([^)]*\)\]', '', text)
    # Remove JSON objects that look like tool results: {...}
    text = re.sub(r'\{[^{}]*\}', '', text)
    return text.strip()


def classify_and_expand_intent(user_message: str) -> str:
    """Add routing hints so the model picks the right tool."""
    lower = user_message.lower()
    
    # 1. Isolate the user's actual prompt to prevent matching injected system notes
    # (which might contain strings like "Lights OFF" and artificially trigger the wrong hint).
    user_part = lower.split("\nuser: ")[-1] if "\nuser: " in lower else lower

    # 2. Avoid overriding single-room commands if it's a relative/bulk request
    is_relative = any(k in user_part for k in {
        "other", "rest", "remaining", "except", "apart"
    })

    if not is_relative:
        # ── Room-context light commands ────────────────────────────────────────────
        # Triggered when server.py has already injected "(System note: The user is
        # currently in the <room>. ...)" into the message.
        room_match = re.search(r'currently in the (\w+)', lower)
        if room_match and any(w in user_part for w in ("light", "lights", "lamp", "bulb")):
            room = room_match.group(1)
            
            # Check user_part specifically to avoid false positives from the system note
            if any(w in user_part for w in ("off", "turn off", "disable", "switch off")):
                return (f"{user_message}\n"
                        f"[HINT: Use toggle_lights(room='{room}', state='off') — "
                        f"room context already resolved, do NOT call intent_unclear.]")
            if any(w in user_part for w in ("on", "turn on", "enable", "switch on")):
                return (f"{user_message}\n"
                        f"[HINT: Use toggle_lights(room='{room}', state='on') — "
                        f"room context already resolved, do NOT call intent_unclear.]")

    # ── Bulk operations ────────────────────────────────────────────────────────
    hints = {
        'all lights':   'Use toggle_all_lights(state=...) — NOT toggle_lights.',
        'all rooms':    'Use toggle_all_lights(state=...) — NOT toggle_lights.',
        'entire house': 'Use toggle_all_lights and/or lock_all_doors for bulk operations.',
        'all doors':    'Use lock_all_doors(state=...) — NOT lock_door.',
        'everything':   'Use bulk ops: toggle_all_lights and lock_all_doors.',
    }
    for kw, hint in hints.items():
        if kw in user_part:
            return f"{user_message}\n[HINT: {hint}]"
    
    # ── Scene commands ─────────────────────────────────────────────────────────
    scene_keywords = {
        'bedtime':     'Use set_scene(scene=\'bedtime\')',
        'movie night': 'Use set_scene(scene=\'movie_night\')',
        'morning':     'Use set_scene(scene=\'morning\')',
        'away':        'Use set_scene(scene=\'away\')',
        'party':       'Use set_scene(scene=\'party\')',
    }
    for scene_kw, scene_hint in scene_keywords.items():
        if scene_kw in user_part or any(w in user_part for w in ['scene', 'mode', 'activate']):
            if any(scene in user_part for scene in ['bedtime', 'morning', 'movie', 'away', 'party']):
                if 'bedtime' in user_part:
                    return f"{user_message}\n[HINT: Use set_scene(scene='bedtime')]"
                elif 'morning' in user_part:
                    return f"{user_message}\n[HINT: Use set_scene(scene='morning')]"
                elif 'movie' in user_part:
                    return f"{user_message}\n[HINT: Use set_scene(scene='movie_night')]"
                elif 'away' in user_part:
                    return f"{user_message}\n[HINT: Use set_scene(scene='away')]"
                elif 'party' in user_part:
                    return f"{user_message}\n[HINT: Use set_scene(scene='party')]"
            
    return user_message
# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    history: list[dict] | None = None,
    backend: str = "local",
    on_tool_call=None,
    messages_out: list | None = None,
    temperature: float = 0.0,
) -> str:
    """Runs the agent loop and returns the final text response."""

    backend_cfg = BACKENDS[backend]
    client = backend_cfg["client"]
    model  = backend_cfg["model"]

    expanded_message = classify_and_expand_intent(user_message)
    print(f"[Agent] Original: {user_message}")
    print(f"[Agent] Expanded: {expanded_message}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": expanded_message},
    ]

    seen_calls: set[str] = set()
    max_iter = 5
    final_response = None
    iteration = 0

    for _ in range(max_iter):
        iteration += 1
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=temperature,
        )
        message = response.choices[0].message
        print(f"[Agent] Model response - tool_calls: {message.tool_calls}, "
              f"content: {(message.content or '')[:120]}")

        # Support both native tool_calls and text-encoded calls from local models
        parsed_tool_calls = (
            parse_tool_calls_from_text(message.content)
            if not message.tool_calls
            else None
        )
        tool_calls_to_use = message.tool_calls or parsed_tool_calls

        if not tool_calls_to_use:
            final_response = clean_text_response(message.content or "")
            
            # On iteration 1, if no tool calls generated, try clearing history and retry once
            if iteration == 1 and history:
                print(f"[Agent] Iteration 1 failed to find tool calls with history present. Clearing history and retrying...")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": expanded_message},
                ]
                continue
            
            messages.append({"role": "assistant", "content": message.content})
            break

        # Wrap text-parsed calls in a mock object matching the OpenAI shape
        if parsed_tool_calls:
            class MockToolCall:
                def __init__(self, d: dict):
                    self.id   = d["id"]
                    self.type = "function"
                    self.function = type("F", (), {
                        "name":      d["name"],
                        "arguments": json.dumps(d["args"]),
                    })()
            tool_calls_to_use = [MockToolCall(tc) for tc in parsed_tool_calls]

        # Duplicate-call guard
        if any(
            f"{tc.function.name}:{json.dumps(json.loads(tc.function.arguments), sort_keys=True)}"
            in seen_calls
            for tc in tool_calls_to_use
        ):
            break

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls_to_use
            ],
        })

        for tool_call in tool_calls_to_use:
            name     = tool_call.function.name
            raw_args = json.loads(tool_call.function.arguments)

            # ── Fix 1: coerce wrong-key hallucinations ─────────────────────
            args = coerce_args(name, raw_args)

            # ── Fix 2 + 3: validate; on error feed back to model ───────────
            error_msg = validate_tool_call(name, args)
            if error_msg:
                print(f"[Validate] {error_msg} — feeding back to model for retry")
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      json.dumps({"error": error_msg}),
                })
                # Don't call the handler; the loop will ask the model to retry
                continue

            # All clear — execute
            call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
            seen_calls.add(call_key)

            handler = TOOL_HANDLERS.get(name)
            result  = handler(**args) if handler else {"error": f"Unknown tool: {name}"}

            if on_tool_call:
                on_tool_call(name, args, result)

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(result),
            })

    if final_response is None:
        # Force a plain-text summary after tool execution or hitting max_iter
        final = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="none",
            temperature=temperature,
            max_tokens=256,
        )
        final_response = final.choices[0].message.content or "Done."
        messages.append({"role": "assistant", "content": final_response})

    if messages_out is not None:
        messages_out.extend(messages)

    return clean_text_response(final_response)


def run_agent_stream(
    user_message: str,
    history: list[dict] | None = None,
    backend: str = "local",
    temperature: float = 0.0,
    messages_out: list | None = None,
):
    """
    Generator version of run_agent.
    Yields dicts:
      {"type": "tool_call", "name": ..., "args": ..., "result": ...}
      {"type": "token",     "text": ...}
      {"type": "done",      "text": <full_final_text>}
      {"type": "error",     "text": ...}
    """
    backend_cfg = BACKENDS[backend]
    client      = backend_cfg["client"]
    model       = backend_cfg["model"]

    expanded_message = classify_and_expand_intent(user_message)
    print(f"[Stream] Initial Prompt: {user_message}")
    if expanded_message != user_message:
        print(f"[Stream] Expanded HINT: {expanded_message}")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": expanded_message},
    ]

    seen_calls: set[str] = set()
    last_text_response = ""

    # Yield status before we send the user prompt to the model
    yield {"type": "status", "text": "Performing action"}

    # ── Tool execution loop (same logic as run_agent, non-streaming) ──────────
    for i in range(5):
        try:
            print(f"[Stream] Iteration {i+1}: Calling LLM for tool/intent detection...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=temperature,
            )
        except Exception as e:
            yield {"type": "error", "text": str(e)}
            return

        message = response.choices[0].message
        if message.content:
            last_text_response += message.content

        parsed_tool_calls = (
            parse_tool_calls_from_text(message.content)
            if not message.tool_calls
            else None
        )
        tool_calls_to_use = message.tool_calls or parsed_tool_calls

        if not tool_calls_to_use:
            # No more tool calls — break out and stream the accumulated content
            print(f"[Stream] No tool calls detected.")
            
            # On iteration 1, if no tool calls generated, try clearing history and retry once
            if i == 0 and history:
                print(f"[Stream] Iteration 1 failed to find tool calls with history present. Clearing history and retrying...")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": expanded_message},
                ]
                continue
            
            messages.append({"role": "assistant", "content": message.content})
            break

        print(f"[Stream] Found {len(tool_calls_to_use)} tool call(s) to execute.")

        # Wrap text-parsed calls in mock objects
        if parsed_tool_calls:
            class MockToolCall:
                def __init__(self, d: dict):
                    self.id   = d["id"]
                    self.type = "function"
                    self.function = type("F", (), {
                        "name":      d["name"],
                        "arguments": json.dumps(d["args"]),
                    })()
            tool_calls_to_use = [MockToolCall(tc) for tc in parsed_tool_calls]

        messages.append({
            "role":    "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id":   tc.id,
                    "type": tc.type,
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls_to_use
            ],
        })

        for tc in tool_calls_to_use:
            name     = tc.function.name
            raw_args = json.loads(tc.function.arguments)
            args     = coerce_args(name, raw_args)

            error_msg = validate_tool_call(name, args)
            if error_msg:
                print(f"[Validate] {error_msg}")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({"error": error_msg}),
                })
                continue

            call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
            if call_key in seen_calls:
                continue
            seen_calls.add(call_key)

            handler = TOOL_HANDLERS.get(name)
            result  = handler(**args) if handler else {"error": f"Unknown tool: {name}"}
            print(f"[Stream] Tool '{name}' result: {result}")

            # ── Emit tool_call event immediately so UI updates in real time ──
            print(f"[Stream] YIELDING tool_call to frontend: {name}({args})")
            yield {"type": "tool_call", "name": name, "args": args, "result": result}
            
            yield {"type": "status", "text": "Action performed..."}

            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    # ── Stream the final plain-text summary ───────────────────────────────────
    full_text = ""
    last_text_response = clean_text_response(last_text_response)

    if last_text_response:
        # If the loop already produced a text response, stream it directly
        # without making another LLM call — this eliminates the redundant 40s call
        for word in last_text_response.split(" "):
            if not word: continue
            token = word + " "
            full_text += token
            yield {"type": "token", "text": token}
    else:
        # Only call the model again if tools were the last thing that ran
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="none",
                temperature=temperature,
                max_tokens=256,
                stream=True,
            )
            print(f"[Stream] Starting final summary generation...")
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_text += delta
                    # print(f"[Stream] YIELDING token: {delta!r}") # Too noisy?
                    yield {"type": "token", "text": delta}

        except Exception as e:
            yield {"type": "error", "text": str(e)}
            return

    full_text = clean_text_response(full_text) or "Done."
    messages.append({"role": "assistant", "content": full_text})

    if messages_out is not None:
        messages_out.extend(messages)

    yield {"type": "done", "text": full_text}
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
    "\n"
    "ROOMS: bedroom, bathroom, office, hallway, kitchen, living_room\n"
    "DOORS: front, back, garage, side\n"
    "\n"
    "TOOLS — use EXACTLY these parameter names:\n"
    "  toggle_lights(room=<room>, state='on'|'off')          — one room\n"
    "  toggle_all_lights(state='on'|'off')                   — ALL rooms at once\n"
    "  lock_door(door=<door>, state='lock'|'unlock')         — one door\n"
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
    Parse tool calls from text like: [toggle_lights(room="kitchen", state="on")]
    Handles both quoted strings and unquoted numeric values.
    """
    tool_calls = []
    pattern = r'\[(\w+)\((.*?)\)\]'
    for idx, match in enumerate(re.finditer(pattern, text)):
        func_name = match.group(1)
        args_str  = match.group(2)
        try:
            args: dict = {}
            # quoted strings
            for key, val in re.findall(r'(\w+)="([^"]*)"', args_str):
                args[key] = val
            # unquoted numbers (e.g. temperature=72)
            for key, val in re.findall(r"(\w+)=([0-9]+(?:\.[0-9]+)?)\b", args_str):
                if key not in args:
                    args[key] = val
            tool_calls.append({"name": func_name, "args": args, "id": f"call_{idx}"})
            print(f"[ParseToolCalls] Extracted: {func_name}({args})")
        except Exception as e:
            print(f"[ParseToolCalls] Failed to parse '{match.group(0)}': {e}")
    return tool_calls


def clean_text_response(text: str) -> str:
    """Remove tool call syntax from a text response."""
    return re.sub(r'\[(\w+)\([^)]*\)\]', '', text).strip()


def classify_and_expand_intent(user_message: str) -> str:
    """Add a routing hint for bulk operations so the model picks the right tool."""
    lower = user_message.lower()
    hints = {
        'all lights':   'Use toggle_all_lights(state=...) — NOT toggle_lights.',
        'all rooms':    'Use toggle_all_lights(state=...) — NOT toggle_lights.',
        'entire house': 'Use toggle_all_lights and/or lock_all_doors for bulk operations.',
        'all doors':    'Use lock_all_doors(state=...) — NOT lock_door.',
        'everything':   'Use bulk ops: toggle_all_lights and lock_all_doors.',
    }
    for kw, hint in hints.items():
        if kw in lower:
            return f"{user_message}\n[HINT: {hint}]"
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

    for _ in range(max_iter):
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
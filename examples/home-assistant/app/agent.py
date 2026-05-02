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

def get_system_prompt(avail_r: list, avail_d: list, tv_str: str, spk_str: str, fan_str: str) -> str:
    return (
        "You are a smart home assistant AI. Use tools to control the home.\n\n"
        "TOOLS:\n"
        "  toggle_lights(room, state='on'|'off')\n"
        "  toggle_all_lights(state='on'|'off')\n"
        "  lock_door(door, state='lock'|'unlock')\n"
        "  lock_all_doors(state='lock'|'unlock')\n"
        "  set_thermostat(temperature=<int>, mode='heat'|'cool'|'auto')\n"
        "  set_scene(scene='movie_night'|'bedtime'|'morning'|'away'|'party')\n"
        "  control_tv(room, state='on'|'off')\n"
        "  control_fan(room, state='on'|'off'[, speed='low'|'medium'|'high'])\n"
        "  control_speaker(room, action='play'|'pause'|'stop'|'next'|'previous')\n"
        "  intent_unclear(reason='off_topic'|'incomplete'|"
        "'unsupported_device'|'unsupported_feature')\n\n"
        f"CONNECTED ROOMS (lights): {', '.join(avail_r)}\n"
        f"CONNECTED DOORS: {', '.join(avail_d)}\n"
        f"CONNECTED TVs: {tv_str}\n"
        f"CONNECTED SPEAKERS: {spk_str}\n"
        f"CONNECTED FANS: {fan_str}\n\n"
        "STATE RULES:\n"
        "  [STATE:] shows all current device states.\n"
        "  State already matches request → plain text reply, NO tool call.\n"
        "  Only rooms listed under CONNECTED TVs/SPEAKERS/FANS have those devices.\n"
        "  Requesting a device in an unlisted room → intent_unclear(unsupported_device).\n\n"
        "TV / SPEAKER / FAN RESOLUTION when user says 'the TV'/'the fan'/'the speaker':\n"
        "  1. Exactly one connected → use that room automatically.\n"
        "  2. Multiple connected + current_user_room has device → use current_user_room.\n"
        "  3. Multiple connected + exactly ONE is in the eligible state for the action\n"
        "     (e.g. only one TV is on and user says 'turn off the TV') → infer that room.\n"
        "  4. Multiple connected + ambiguous (rule 2 & 3 don't apply) → intent_unclear(incomplete).\n\n"
        "LIGHT / DOOR RESOLUTION:\n"
        "  current_user_room set + connected → use current_user_room.\n"
        "  current_user_room set + NOT connected → intent_unclear(unsupported_device).\n"
        "  current_user_room empty → intent_unclear(incomplete).\n\n"
        "  [RECENT ACTIONS:] → resolve 'undo', 'again', 'same for X', and pronouns ('it').\n"
        "  SYNONYMS: 'open'='unlock'; 'close'/'shut'='lock'; 'skip'='next'; 'back'='previous'."
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
    Parse tool calls from model output text.

    Priority order:
      1. <|tool_call_start|>{"name":..., "parameters":...}<|tool_call_end|>
         (the format the local model was trained on)
      2. Bare JSON objects with "name" key (llama-server strips special tokens)
      3. func_name(key="val", ...) regex fallback (for OpenAI text leakage)
    """
    if not text:
        return []

    tool_calls = []

    valid_tools = {
        "toggle_lights", "toggle_all_lights", "lock_door",
        "lock_all_doors", "set_thermostat", "set_scene", "intent_unclear",
        "control_tv", "control_speaker", "control_fan"
    }

    # ── Priority 1: custom token format (training format) ─────────────────────
    blocks = re.findall(
        r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>', text, re.DOTALL)
    if blocks:
        for idx, block in enumerate(blocks):
            try:
                obj = json.loads(block.strip())
                name = obj.get("name", "")
                args = obj.get("parameters", obj.get("arguments", {}))
                tool_calls.append({"name": name, "args": args, "id": f"call_{idx}"})
                print(f"[ParseToolCalls] Token format: {name}({args})")
            except Exception as e:
                print(f"[ParseToolCalls] Failed to parse token block: {e}")
        if tool_calls:
            return tool_calls

    # ── Priority 2: bare JSON objects (llama-server strips special tokens) ────
    #    Model outputs: {"name": "tool", "parameters": {"room": "x", "state": "y"}}
    #    The regex handles one level of nested braces for the parameters dict.
    json_pattern = r'\{"name"\s*:\s*"(\w+)"[^}]*"(?:parameters|arguments)"\s*:\s*(\{[^}]*\})[^}]*\}'
    for idx, match in enumerate(re.finditer(json_pattern, text)):
        try:
            name = match.group(1)
            if name not in valid_tools:
                continue
            args = json.loads(match.group(2))
            tool_calls.append({"name": name, "args": args, "id": f"call_{idx}"})
            print(f"[ParseToolCalls] JSON format: {name}({args})")
        except (json.JSONDecodeError, Exception) as e:
            print(f"[ParseToolCalls] JSON parse error: {e}")
    if tool_calls:
        return tool_calls

    # ── Priority 2: regex fallback for func(args) style ───────────────────────
    valid_tools = {
        "toggle_lights", "toggle_all_lights", "lock_door",
        "lock_all_doors", "set_thermostat", "set_scene", "intent_unclear",
        "control_tv", "control_speaker", "control_fan"
    }

    pattern = r'(\w+)\(([^)]*)\)'
    for idx, match in enumerate(re.finditer(pattern, text)):
        func_name = match.group(1)
        args_str  = match.group(2)

        if func_name not in valid_tools:
            continue

        try:
            args: dict = {}
            for key, val in re.findall(r'(\w+)=["\']([^"\']*)["\']', args_str):
                args[key] = val
            for key, val in re.findall(r"(\w+)=([0-9]+(?:\.[0-9]+)?)\b", args_str):
                if key not in args:
                    args[key] = val

            tool_calls.append({"name": func_name, "args": args, "id": f"call_{idx}"})
            print(f"[ParseToolCalls] Regex format: {func_name}({args})")
        except Exception as e:
            print(f"[ParseToolCalls] Failed to parse '{match.group(0)}': {e}")

    return tool_calls

def extract_think_and_response(text: str) -> tuple[str, str]:
    """
    Extracts the think trace and the actual response from the raw output.
    Handles cases where special tokens (<think>, <|tool_call_start|>) are stripped.
    """
    think_text = ""
    response_text = text

    # 1. Try proper token matching
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
        response_text = text[:think_match.start()] + text[think_match.end():]
        return think_text, response_text

    # 2. Heuristic extraction for stripped tokens
    # Model outputs: [think trace] {"name": "...", "parameters": {...}} [response text]
    valid_tools = {
        "toggle_lights", "toggle_all_lights", "lock_door",
        "lock_all_doors", "set_thermostat", "set_scene", "intent_unclear",
        "control_tv", "control_speaker", "control_fan"
    }

    # Find the first JSON tool call pattern to split on
    json_pattern = r'\{"name"\s*:\s*"(\w+)"[^}]*"(?:parameters|arguments)"\s*:\s*\{[^}]*\}[^}]*\}'
    match = re.search(json_pattern, text)
    if match and match.group(1) in valid_tools:
        think_text = text[:match.start()].strip()
        response_text = text[match.end():].strip()
    else:
        # Fallback for regex text-based tool calls
        tool_pattern = r'(?:[\(\[]?)\b(' + '|'.join(valid_tools) + r')\([^)]*\)(?:[\)\]]?)'
        match = re.search(tool_pattern, text)
        if match:
            think_text = text[:match.start()].strip()
            response_text = text[match.end():].strip()
        else:
            response_text = text

    # Clean up known decoding artifacts (<think> and </think> decode to fjärilsart)
    think_text = think_text.replace("fjärilsart", "").strip()
    response_text = response_text.replace("fjärilsart", "").strip()

    # Clean up any dangling braces that might be left over from splits
    think_text = re.sub(r'[\{\}]+$', '', think_text).strip()
    response_text = re.sub(r'^[\{\}\s,]+', '', response_text).strip()

    return think_text, response_text

def clean_text_response(text: str) -> str:
    """Remove remaining special tokens and syntax from text."""
    # 0. Remove <think>...</think> blocks if any survived
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 1. Remove <|tool_call_start|>...<|tool_call_end|> blocks
    text = re.sub(r'<\|tool_call_start\|>.*?<\|tool_call_end\|>', '', text, flags=re.DOTALL)

    # 2. Remove any remaining special tokens
    text = re.sub(r'<\|[^|]*\|>', '', text)

    valid_tools = {
        "toggle_lights", "toggle_all_lights", "lock_door",
        "lock_all_doors", "set_thermostat", "set_scene", "intent_unclear",
        "control_tv", "control_speaker", "control_fan"
    }
    # 3. Remove tool call syntax like [func(args)]
    tool_pattern = r'(?:[\(\[]?)\b(' + '|'.join(valid_tools) + r')\([^)]*\)(?:[\)\]]?)'
    text = re.sub(tool_pattern, '', text)

    # 4. Strip leftover JSON objects (handles one level of nested braces for parameters)
    json_pattern = r'\{"name"\s*:\s*"[^"]*"[^}]*"(?:parameters|arguments)"\s*:\s*\{[^}]*\}[^}]*\}'
    text = re.sub(json_pattern, '', text)

    # 5. Final cleanup of excess whitespace and dangling artifact chars at start
    text = re.sub(r'^[\{\}\s,]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent_stream(
    user_message: str,
    system_prompt: str,
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

    user_message should already contain [STATE: ...] and [RECENT ACTIONS: ...]
    prefixes injected by the server layer.

    LOCAL BACKEND:
      The fine-tuned model outputs everything in a single turn:
        <think>...</think>
        <|tool_call_start|>{"name":..., "parameters":...}<|tool_call_end|>
        Response text
      So we make ONE API call (no tools param — the system prompt already
      describes tools), parse tool calls from the text, execute them, and
      stream the response text.

    OPENAI BACKEND:
      Uses the standard multi-turn tool execution loop with structured
      tool_calls and tool results.
    """
    backend_cfg = BACKENDS[backend]
    client      = backend_cfg["client"]
    model       = backend_cfg["model"]

    print(f"[Stream] Prompt: {user_message[:200]}...")

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  LOCAL BACKEND — single-turn, TRUE STREAMING
    # ══════════════════════════════════════════════════════════════════════════
    #
    # The model output (after llama-server strips special tokens) follows
    # this layout:
    #   [thinking text] {"name":..., "parameters":...} [response text]
    #
    # We stream with stream=True and use a real-time state machine to:
    #   1. Yield pre-tool text as  {"type": "token"}  (the frontend will
    #      promote it to a thinking block when a tool_call event arrives).
    #   2. Buffer tool-call JSON silently (brace counting).
    #   3. Parse + execute tool calls immediately when complete.
    #   4. Yield post-tool text as  {"type": "token"}.
    #
    # If no tool call is found, all tokens are the final response.
    # ══════════════════════════════════════════════════════════════════════════
    if backend == "local":
        try:
            print(f"[Stream-Local] Calling LLM (single-turn, stream=True)...")
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
                stream=True,
            )
        except Exception as e:
            yield {"type": "error", "text": str(e)}
            return

        # ── Streaming state machine ───────────────────────────────────────
        full_output   = ""          # complete raw output for logging
        pending       = ""          # text waiting to be emitted as tokens
        tool_buffer   = ""          # accumulates JSON while in TOOL state
        brace_depth   = 0
        state         = "streaming" # "streaming" | "tool_buffer"
        tool_events   = []
        pre_tool_text = ""          # all text emitted before first tool call
        saw_tool_call = False
        artifacts     = {"fjärilsart"}  # decoded <think>/<think> artifacts

        FLUSH_THRESHOLD = 3         # flush pending tokens every N chars

        for chunk in stream:
            delta = ""
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
            if not delta:
                continue

            full_output += delta

            for char in delta:
                # ── Filter known decoding artifacts ───────────────────────
                # (handled after full accumulation below)

                if state == "streaming":
                    if char == "{":
                        # Potential tool-call JSON start — flush pending, switch
                        if pending:
                            # Clean artifact strings from pending before emitting
                            cleaned = pending
                            for art in artifacts:
                                cleaned = cleaned.replace(art, "")
                            if cleaned:
                                if not saw_tool_call:
                                    pre_tool_text += cleaned
                                yield {"type": "token", "text": cleaned}
                            pending = ""
                        state = "tool_buffer"
                        tool_buffer = "{"
                        brace_depth = 1
                    else:
                        pending += char
                        # Flush at word boundaries or threshold for responsiveness
                        if char in " \n" or len(pending) >= FLUSH_THRESHOLD:
                            cleaned = pending
                            for art in artifacts:
                                cleaned = cleaned.replace(art, "")
                            if cleaned:
                                if not saw_tool_call:
                                    pre_tool_text += cleaned
                                yield {"type": "token", "text": cleaned}
                            pending = ""

                elif state == "tool_buffer":
                    tool_buffer += char
                    if char == "{":
                        brace_depth += 1
                    elif char == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            # Complete JSON object — is it a tool call?
                            if '"name"' in tool_buffer:
                                saw_tool_call = True
                                # Tell frontend pre-tool text was thinking
                                if pre_tool_text.strip():
                                    yield {"type": "think_done"}
                                # Parse and execute immediately
                                parsed = parse_tool_calls_from_text(tool_buffer)
                                for tc in parsed:
                                    name     = tc["name"]
                                    raw_args = tc["args"]
                                    args     = coerce_args(name, raw_args)

                                    error_msg = validate_tool_call(name, args)
                                    if error_msg:
                                        print(f"[Validate] {error_msg}")
                                        continue

                                    handler = TOOL_HANDLERS.get(name)
                                    result  = handler(**args) if handler else {"error": f"Unknown tool: {name}"}
                                    print(f"[Stream-Local] Tool '{name}' result: {result}")

                                    tool_events.append({"name": name, "args": args, "result": result})
                                    yield {"type": "tool_call", "name": name, "args": args, "result": result}
                            else:
                                # Not a tool call — just a stray JSON object, emit as text
                                cleaned = tool_buffer
                                for art in artifacts:
                                    cleaned = cleaned.replace(art, "")
                                if cleaned:
                                    if not saw_tool_call:
                                        pre_tool_text += cleaned
                                    yield {"type": "token", "text": cleaned}
                            tool_buffer = ""
                            state = "streaming"
                    # Safety valve: if tool_buffer gets huge, it's not a tool call
                    elif len(tool_buffer) > 600:
                        cleaned = tool_buffer
                        for art in artifacts:
                            cleaned = cleaned.replace(art, "")
                        if cleaned:
                            if not saw_tool_call:
                                pre_tool_text += cleaned
                            yield {"type": "token", "text": cleaned}
                        tool_buffer = ""
                        state = "streaming"

        # ── Flush any remaining buffered text ─────────────────────────────
        if pending:
            cleaned = pending
            for art in artifacts:
                cleaned = cleaned.replace(art, "")
            if cleaned:
                yield {"type": "token", "text": cleaned}
        if tool_buffer:
            # Incomplete tool buffer at end of stream — emit as text
            cleaned = tool_buffer
            for art in artifacts:
                cleaned = cleaned.replace(art, "")
            if cleaned:
                yield {"type": "token", "text": cleaned}

        print(f"[Stream-Local] Raw output: {full_output[:300]}")
        print(f"[Stream-Local] Tool calls found: {len(tool_events)}")

        # ── Build final text from the cleaned full output ─────────────────
        _, raw_response = extract_think_and_response(full_output)
        final_text = clean_text_response(raw_response) or "Done."
        messages.append({"role": "assistant", "content": final_text})

        if messages_out is not None:
            messages_out.extend(messages)

        yield {"type": "done", "text": final_text}
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  OPENAI BACKEND — multi-turn tool execution loop
    # ══════════════════════════════════════════════════════════════════════════
    seen_calls: set[str] = set()
    last_text_response = ""

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
            print(f"[Stream] No tool calls detected.")
            messages.append({"role": "assistant", "content": message.content})
            break

        print(f"[Stream] Found {len(tool_calls_to_use)} tool call(s) to execute.")

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

            print(f"[Stream] YIELDING tool_call to frontend: {name}({args})")
            yield {"type": "tool_call", "name": name, "args": args, "result": result}

            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    # ── Stream the final plain-text summary ───────────────────────────────────
    full_text = ""
    last_text_response = clean_text_response(last_text_response)

    if last_text_response:
        for word in last_text_response.split(" "):
            if not word: continue
            token = word + " "
            full_text += token
            yield {"type": "token", "text": token}
    else:
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
                    yield {"type": "token", "text": delta}

        except Exception as e:
            yield {"type": "error", "text": str(e)}
            return

    full_text = clean_text_response(full_text) or "Done."
    messages.append({"role": "assistant", "content": full_text})

    if messages_out is not None:
        messages_out.extend(messages)

    yield {"type": "done", "text": full_text}
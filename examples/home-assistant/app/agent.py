#app/agent.py
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from app.tools.schemas import TOOL_SCHEMAS
from app.tools.handlers import TOOL_HANDLERS

load_dotenv()

local_client  = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

BACKENDS = {
    "local":  {"client": local_client,  "model": "local"},
    "openai": {"client": openai_client, "model": "gpt-4o-mini"},
}

SYSTEM_PROMPT = (
    "You are a home assistant AI. Use tools to control the home; respond in text when no tool is needed. "
    "Output function calls as JSON.\n"
    "INDIVIDUAL CONTROL:\n"
    "- Lights (on/off): bedroom, bathroom, office, hallway, kitchen, living_room.\n"
    "- Doors (lock/unlock): front, back, garage, side.\n"
    "- Thermostat: temperature 60-80°F, modes: heat, cool, auto.\n"
    "BULK CONTROL (for 'all' or 'entire house'):\n"
    "- toggle_all_lights: turn all lights in the house on/off at once.\n"
    "- lock_all_doors: lock/unlock all doors at once.\n"
    "PRESETS:\n"
    "- Scenes: movie_night, bedtime, morning, away, party.\n"
    "GUIDELINES:\n"
    "- For 'turn on all lights' or 'turn off all lights', use toggle_all_lights.\n"
    "- For 'lock all doors' or 'unlock all doors', use lock_all_doors.\n"
    "- For 'turn on lights in [room]', use toggle_lights with that room.\n"
    "- Call intent_unclear (never plain text) when the request is: "
    "ambiguous (could be satisfied by multiple different home control actions, e.g. 'make it nicer in here' could mean thermostat, lights, or a scene), "
    "off_topic (unrelated to home control), "
    "incomplete (no target device or room specified even after reading conversation history, e.g. 'turn it on' as the opening message), "
    "or unsupported_device (refers to a device or feature not available, e.g. brightness, TV, music)."
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
    Parse tool calls from text output like: [toggle_lights(room="kitchen", state="on")]
    Returns a list of dicts with tool_call info compatible with server expectations.
    """
    tool_calls = []
    # Match pattern: [function_name(args)]
    pattern = r'\[(\w+)\((.*?)\)\]'
    matches = re.finditer(pattern, text)
    
    for idx, match in enumerate(matches):
        func_name = match.group(1)
        args_str = match.group(2)
        
        try:
            # Parse arguments like: room="kitchen", state="on"
            args = {}
            # Split by comma, but be careful with quoted strings
            arg_pairs = re.findall(r'(\w+)="([^"]*)"', args_str)
            for key, value in arg_pairs:
                args[key] = value
            
            tool_calls.append({
                "name": func_name,
                "args": args,
                "id": f"call_{idx}",  # Synthetic ID since model didn't provide one
            })
            print(f"[ParseToolCalls] Extracted: {func_name}({args})")
        except Exception as e:
            print(f"[ParseToolCalls] Failed to parse '{match.group(0)}': {e}")
    
    return tool_calls


def clean_text_response(text: str) -> str:
    """Remove tool call syntax from text response."""
    # Remove patterns like [function_name(...)]
    cleaned = re.sub(r'\[(\w+)\([^)]*\)\]', '', text)
    return cleaned.strip()


def classify_and_expand_intent(user_message: str) -> str:
    """
    Analyze user request to provide helpful context for bulk operations.
    This helps the model understand requests like 'turn on all lights'.
    """
    lower_msg = user_message.lower()
    
    # Detect bulk operation keywords
    bulk_keywords = {
        'all lights': 'Use toggle_all_lights to control all lights at once.',
        'all rooms': 'Use toggle_all_lights to control lights in all rooms at once.',
        'entire house': 'Use toggle_all_lights and lock_all_doors for bulk operations.',
        'all doors': 'Use lock_all_doors to control all doors at once.',
        'everything': 'You can use bulk operations: toggle_all_lights and lock_all_doors.',
    }
    
    # Check for bulk operation keywords
    for keyword, hint in bulk_keywords.items():
        if keyword in lower_msg:
            # Return expanded prompt with hint
            return f"{user_message}\n[HINT: {hint}]"
    
    return user_message


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

    # Classify and expand user intent to help model understand bulk operations
    expanded_message = classify_and_expand_intent(user_message)
    print(f"[Agent] Original: {user_message}")
    print(f"[Agent] Expanded: {expanded_message}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": expanded_message},
    ]

    seen_calls: set[str] = set()  # Guard against repeated identical tool calls
    max_iter = 5
    final_response = None
    for _ in range(max_iter):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=temperature,
            # max_tokens=512,
        )
        message = response.choices[0].message
        print(f"[Agent] Model response - tool_calls: {message.tool_calls}, content: {message.content[:100]}")

        # BACKWARD COMPATIBILITY: Handle both formats
        # - Current (text parsing): LFM model outputs text like "[toggle_lights(room="kitchen", state="on")]..."
        # - Future (structured): Fine-tuned model returns proper OpenAI tool_calls format
        parsed_tool_calls = parse_tool_calls_from_text(message.content) if not message.tool_calls else None
        tool_calls_to_use = message.tool_calls or parsed_tool_calls

        if not tool_calls_to_use:
            # No tool calls found; clean any tool syntax from response and use as final
            final_response = clean_text_response(message.content)
            messages.append({"role": "assistant", "content": message.content})
            break

        # If we parsed tool calls from text, convert to the expected tool_call object format
        # This adapter ensures both text-based and structured tool calls work uniformly
        if parsed_tool_calls:
            # For parsed calls, simulate the tool_call structure that would come from a structured model
            class MockToolCall:
                def __init__(self, call_dict):
                    self.id = call_dict["id"]
                    self.type = "function"
                    self.function = type('obj', (object,), {
                        'name': call_dict["name"],
                        'arguments': json.dumps(call_dict["args"])
                    })()
            
            tool_calls_to_use = [MockToolCall(tc) for tc in parsed_tool_calls]

        # Check for duplicate calls before appending to messages, so the
        # messages list stays in a valid state for the forced-text fallback.
        duplicate = any(
            f"{tc.function.name}:{json.dumps(json.loads(tc.function.arguments), sort_keys=True)}"
            in seen_calls
            for tc in tool_calls_to_use
        )
        if duplicate:
            break

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls_to_use
            ],
        })

        for tool_call in tool_calls_to_use:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
            seen_calls.add(call_key)

            handler = TOOL_HANDLERS.get(name)
            result = handler(**args) if handler else {"error": f"Unknown tool: {name}"}

            if on_tool_call:
                on_tool_call(name, args, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    if final_response is None:
        # Forced text-only call: model summarises what it just did.
        # Reached when the model loops on duplicate tool calls or hits max_iter.
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

    # Clean the response text of any tool syntax before returning
    return clean_text_response(final_response)

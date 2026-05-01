#app/tools/schemas.py
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "toggle_lights",
            "description": "Turn lights on or off in a specific room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room":  {"type": "string", "enum": ["living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"]},
                    "state": {"type": "string", "enum": ["on", "off"]},
                },
                "required": ["room", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "description": "Set the thermostat temperature (60-80 F) and operating mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "integer", "minimum": 60, "maximum": 80},
                    "mode":        {"type": "string", "enum": ["heat", "cool", "auto"]},
                },
                "required": ["temperature", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_door",
            "description": "Lock or unlock a specific door.",
            "parameters": {
                "type": "object",
                "properties": {
                    "door":  {"type": "string", "enum": ["front", "back", "garage", "side", "bedroom", "bathroom", "office", "kitchen", "living_room"]},
                    "state": {"type": "string", "enum": ["lock", "unlock"]},
                },
                "required": ["door", "state"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "set_scene",
            "description": "Activate a preset home scene that adjusts multiple devices at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "enum": ["movie_night", "bedtime", "morning", "away", "party"]},
                },
                "required": ["scene"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_all_lights",
            "description": "Turn all lights in the entire house on or off at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["on", "off"]},
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_all_doors",
            "description": "Lock or unlock all doors in the house at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["lock", "unlock"]},
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_tv",
            "description": "Turn a TV on or off in a specific room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room":  {"type": "string", "enum": ["living_room", "bedroom", "office"]},
                    "state": {"type": "string", "enum": ["on", "off"]}
                },
                "required": ["room", "state"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_speaker",
            "description": "Control a speaker in a specific room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room":   {"type": "string", "enum": ["living_room", "bedroom", "kitchen", "office", "hallway"]},
                    "action": {"type": "string", "enum": ["play", "pause", "stop", "next", "previous"]}
                },
                "required": ["room", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_fan",
            "description": "Turn the fan on or off in a specific room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room":  {"type": "string", "enum": ["living_room", "bedroom", "kitchen", "office"]},
                    "state": {"type": "string", "enum": ["on", "off"]},
                    "speed": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["room", "state"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "intent_unclear",
            "description": (
                "Call this tool instead of responding with text when the request cannot be fulfilled. "
                "Reason 'ambiguous': request could mean multiple things. "
                "Reason 'off_topic': completely outside home automation (ordering food, weather, music, etc.). "
                "Reason 'incomplete': pronoun or reference with no prior context (e.g. 'turn it on'). "
                "Reason 'unsupported_device': within home domain but feature unavailable (brightness, cameras)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": ["off_topic", "incomplete", "unsupported_device", "unsupported_feature"]},
                },
                "required": ["reason"],
            },
        },
    },
]

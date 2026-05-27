import sys
import os

# Mock ROOM_DISPLAY for testing if needed, but we'll try to import or replicate the logic
ROOM_DISPLAY = {
    "living_room": "living room",
    "bedroom":     "bedroom",
    "kitchen":     "kitchen",
    "bathroom":    "bathroom",
    "office":      "office",
    "hallway":     "hallway",
}

def _room_hint(room: str) -> str:
    display = ROOM_DISPLAY.get(room, room.replace("_", " "))
    return (
        f"Room context: {room}. "
        f"If they ask to control 'the light', assume they mean the {display}. "
        f"If they ask to control 'the door' or 'this door', assume they mean the {display} door."
    )

def test():
    rooms = ["living_room", "bedroom", "kitchen", "bathroom", "office", "hallway", "garage"]
    for r in rooms:
        print(f"Room: {r}")
        print(f"Hint: {_room_hint(r)}")
        print("-" * 20)

if __name__ == "__main__":
    test()

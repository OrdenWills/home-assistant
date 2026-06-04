# app/events.py
import asyncio
import json
import threading

class EventBus:
    """Thread-safe event bus for broadcasting to SSE clients."""
    
    def __init__(self):
        self.listeners: list[asyncio.Queue] = []
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        
    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop (called at startup)."""
        self.loop = loop
        
    async def subscribe(self) -> asyncio.Queue:
        """Create a new event queue for an SSE client."""
        queue = asyncio.Queue(maxsize=100)
        with self.lock:
            self.listeners.append(queue)
        return queue
        
    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a client's event queue."""
        with self.lock:
            if queue in self.listeners:
                self.listeners.remove(queue)
                
    def emit(self, event_type: str, data: dict) -> None:
        """Broadcast event to all clients (thread-safe, callable from any thread)."""
        if not self.loop:
            print(f"[EventBus] Event loop not set. Dropping event: {event_type}")
            return
            
        if not self.validate_event(event_type, data):
            print(f"[EventBus] Validation failed. Dropping event: {event_type}")
            return
            
        asyncio.run_coroutine_threadsafe(
            self._broadcast(event_type, data),
            self.loop
        )
        
    async def _broadcast(self, event_type: str, data: dict) -> None:
        """Internal async broadcast to all queues."""
        message = {"event": event_type, "data": data}
        with self.lock:
            dead_queues = []
            for queue in self.listeners:
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    dead_queues.append(queue)
            for queue in dead_queues:
                if queue in self.listeners:
                    self.listeners.remove(queue)
                    
    def validate_event(self, event_type: str, data: dict) -> bool:
        """Validate event data structure and values."""
        allowed_types = {"track_changed", "playback_state_changed", "repeat_mode_changed", "connection_init"}
        if event_type not in allowed_types:
            print(f"[EventBus] Invalid event type: {event_type}")
            return False
            
        if "state" in data:
            if data["state"] not in {"playing", "paused", "stopped"}:
                print(f"[EventBus] Invalid state: {data['state']}")
                return False
                
        if "track_index" in data:
            val = data["track_index"]
            if not isinstance(val, int) or val < 0:
                print(f"[EventBus] Invalid track_index: {val}")
                return False
                
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            print(f"[EventBus] JSON serialization failed: {e}")
            return False
            
        return True

event_bus = EventBus()

def emit_state_event(event_type: str, data: dict) -> None:
    """Helper to emit events from synchronous code."""
    event_bus.emit(event_type, data)

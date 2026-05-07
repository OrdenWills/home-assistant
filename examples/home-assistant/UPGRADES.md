# Future Upgrades & Roadmap

This document tracks potential improvements, architectural changes, and "situations" encountered during development that warrant future attention.



## Questions:
**Situation:**In the real world what do i want to happen when i say on the speaker when am in the bathroom ? do i want the hallway(which share boundary with all the rooms) speaker to play when there is no speaker in the bathroom.


## Proposed Upgrades

### 1. Action Log Limit & Multi-Turn Reasoning (Done used transaction logging)
**Situation:** When the user issues a command that triggers multiple tool calls (e.g., "turn off all the lights that are on"), the system currently logs each tool call as a separate entry in the `action_log`. 
**Current State:** `MAX_ACTION_LOG` is set to 3. If 4 lights are toggled, the first one is pushed out of the reasoning context for the next turn.
**Proposed Fix:** 
- Increase `MAX_ACTION_LOG` to a higher value (e.g., 10-15).
- OR: Implement "Transaction Logging" where all tool calls from a single user turn are grouped into one log entry.

### 2. Tokenizer Artifact Cleanup
**Situation:** The local inference server (llama-server) sometimes fails to strip special tokens or leaves fragments of JSON/tool-call tokens in the final response.
**Current State:** Heuristic regex patterns in `agent.py` clean up `<think>` tags and JSON artifacts.
**Proposed Fix:** Use a more robust token-based parser if the inference backend supports it, or implement a state-machine parser for tool calls to prevent leaking braces `}}` into the UI.

### 3. Voice Input Enhancements
**Situation:** Basic Web Speech API implementation.
**Proposed Fix:** Add wake-word detection ("Hey Assistant") and better noise cancellation handling.

### 4. Improve handling user location currently the thoughts have things like user location is ''

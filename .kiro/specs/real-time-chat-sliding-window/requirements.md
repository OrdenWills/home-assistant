# Requirements Document

## Introduction

This document specifies the requirements for a real-time sliding window chat pagination system. The system maintains a fixed number of conversation turns visible in the DOM during active chatting, automatically removing the oldest messages as new ones arrive, while preserving the ability to load and view older historical messages through the existing "Load More" functionality.

## Glossary

- **Chat_System**: The frontend chat interface that displays conversation messages
- **Conversation_Turn**: A complete exchange consisting of a user message and the corresponding assistant response (including optional thought process and tool calls)
- **Live_Window**: The most recent conversation turns that are actively maintained in the DOM
- **Historical_Messages**: Older conversation turns that have been loaded via the "Load More" button
- **DOM**: Document Object Model, the in-memory representation of HTML elements
- **Message_Container**: An HTML element with class `msg-container` that holds a single message (user or assistant)

## Requirements

### Requirement 1: Sliding Window Management

**User Story:** As a user engaged in an active conversation, I want only the most recent conversation turns to remain visible, so that the interface stays performant and focused on current context.

#### Acceptance Criteria

1. WHEN the Chat_System contains more than 7 conversation turns in the Live_Window, THE Chat_System SHALL remove the oldest conversation turn from the DOM
2. WHEN counting conversation turns, THE Chat_System SHALL count each user message plus its corresponding assistant response (including thought and tool calls) as one turn
3. WHEN a new assistant response completes, THE Chat_System SHALL evaluate whether the Live_Window exceeds 7 turns and trigger removal if necessary
4. WHEN removing old conversation turns, THE Chat_System SHALL remove all associated elements including user message, thought block, assistant response, and tool calls
5. THE Chat_System SHALL maintain exactly 7 or fewer conversation turns in the Live_Window at all times during active chatting

### Requirement 2: Historical Message Preservation

**User Story:** As a user reviewing conversation history, I want messages loaded via "Load More" to remain visible, so that I can reference older context without it disappearing.

#### Acceptance Criteria

1. WHEN a user clicks the "Load More" button, THE Chat_System SHALL load older messages and insert them at the beginning of the message container
2. WHEN Historical_Messages are present in the DOM, THE Chat_System SHALL NOT count them toward the 7-turn limit
3. WHEN the sliding window removes messages, THE Chat_System SHALL only remove messages from the Live_Window
4. THE Chat_System SHALL distinguish Historical_Messages from Live_Window messages using the "— restored history —" divider as a boundary marker
5. WHEN Historical_Messages are present, THE Chat_System SHALL preserve all messages above the history divider during sliding window operations

### Requirement 3: Visual Separation

**User Story:** As a user, I want clear visual indication of what messages are historical versus live, so that I understand the context of what I'm viewing.

#### Acceptance Criteria

1. WHEN Historical_Messages are loaded, THE Chat_System SHALL display a divider element with class "history-divider" containing the text "— restored history —"
2. THE Chat_System SHALL position the history divider immediately after the last Historical_Message and before the first Live_Window message
3. WHEN all Historical_Messages are removed from view, THE Chat_System SHALL remove the history divider
4. THE Chat_System SHALL maintain the existing visual styling for the history divider

### Requirement 4: Scroll Position Management

**User Story:** As a user, I want the chat interface to maintain appropriate scroll position when messages are removed, so that my reading experience is not disrupted.

#### Acceptance Criteria

1. WHEN the Chat_System removes old conversation turns from the Live_Window, THE Chat_System SHALL preserve the user's current scroll position relative to visible content
2. WHEN a new message arrives and the user is scrolled to the bottom, THE Chat_System SHALL automatically scroll to show the new message
3. WHEN a new message arrives and the user is scrolled up reviewing history, THE Chat_System SHALL NOT automatically scroll to the bottom
4. WHEN Historical_Messages are loaded via "Load More", THE Chat_System SHALL adjust scroll position to maintain the user's view of the content they were reading

### Requirement 5: Performance and Timing

**User Story:** As a user, I want message removal to happen instantly without delays, so that the interface feels responsive and smooth.

#### Acceptance Criteria

1. WHEN the Chat_System removes old conversation turns, THE removal SHALL occur synchronously without setTimeout or delay
2. WHEN evaluating whether to remove messages, THE Chat_System SHALL perform the check immediately after a new assistant response completes
3. THE Chat_System SHALL complete all DOM removal operations within a single event loop tick
4. WHEN removing multiple elements (user message, thought, assistant response, tool calls), THE Chat_System SHALL remove them in a single batch operation

### Requirement 6: Message Counting Logic

**User Story:** As a developer, I want accurate conversation turn counting, so that the sliding window maintains exactly 7 turns as specified.

#### Acceptance Criteria

1. WHEN counting turns in the Live_Window, THE Chat_System SHALL identify turn boundaries by locating user message containers
2. WHEN a user message container is found, THE Chat_System SHALL count it and all subsequent elements until the next user message as one turn
3. THE Chat_System SHALL exclude the history divider element from turn counting
4. THE Chat_System SHALL exclude Historical_Messages (elements before the history divider) from turn counting
5. WHEN the Live_Window contains exactly 7 turns, THE Chat_System SHALL NOT remove any messages

### Requirement 7: Integration with Existing Features

**User Story:** As a user, I want all existing chat features to continue working, so that the sliding window enhancement doesn't break current functionality.

#### Acceptance Criteria

1. WHEN messages are removed by the sliding window, THE Chat_System SHALL preserve feedback buttons (thumbs up/down) on all visible messages
2. WHEN messages are removed by the sliding window, THE Chat_System SHALL preserve rating data on all visible messages
3. THE Chat_System SHALL maintain the existing "Load More" button functionality without modification
4. THE Chat_System SHALL maintain the existing message streaming functionality without modification
5. WHEN the page is refreshed, THE Chat_System SHALL load the last 7 conversation turns as it currently does

### Requirement 8: Edge Case Handling

**User Story:** As a user, I want the system to handle edge cases gracefully, so that the chat interface remains stable under all conditions.

#### Acceptance Criteria

1. WHEN the Chat_System has fewer than 7 conversation turns in the Live_Window, THE Chat_System SHALL NOT attempt to remove any messages
2. WHEN rapid consecutive messages arrive, THE Chat_System SHALL handle each message removal evaluation independently
3. WHEN the history divider is not present, THE Chat_System SHALL treat all messages as Live_Window messages
4. WHEN Historical_Messages are present and new messages arrive, THE Chat_System SHALL maintain the separation between historical and live content
5. IF the DOM structure is malformed or unexpected, THE Chat_System SHALL fail gracefully without throwing errors

### Requirement 9: Parser and Serializer Requirements

**User Story:** As a developer, I want to correctly identify and parse conversation turn structures in the DOM, so that the sliding window can accurately count and remove turns.

#### Acceptance Criteria

1. THE Turn_Parser SHALL identify conversation turns by parsing the DOM structure starting from user message containers
2. THE Turn_Parser SHALL extract all elements belonging to a single turn (user message, thought, assistant response, tool calls)
3. FOR ALL valid conversation turn structures in the DOM, parsing the turn structure SHALL correctly identify all associated elements
4. WHEN a turn structure is parsed, THE Turn_Parser SHALL return a list of DOM elements that comprise the complete turn
5. THE Turn_Parser SHALL handle turns with optional elements (thought blocks, tool calls) correctly

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from time import time

@dataclass
class Message:
    role: str   # "system" | "user" | "assistant" | "tool"
    content: str

class InMemoryChatStore:
    """
    Simple per-session chat memory.
    - Stores last N messages per session
    - Not persistent (restarts wipe memory)
    """
    def __init__(self, max_messages: int = 30):
        self.max_messages = max_messages
        self._sessions: Dict[str, List[Message]] = {}
        self._last_seen: Dict[str, float] = {}

    def get_history(self, session_id: str) -> List[Message]:
        self._last_seen[session_id] = time()
        return self._sessions.get(session_id, []).copy()

    def append(self, session_id: str, role: str, content: str) -> None:
        self._last_seen[session_id] = time()
        msgs = self._sessions.setdefault(session_id, [])
        msgs.append(Message(role=role, content=content))
        if len(msgs) > self.max_messages:
            del msgs[: len(msgs) - self.max_messages]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._last_seen.pop(session_id, None)

store = InMemoryChatStore(max_messages=40)

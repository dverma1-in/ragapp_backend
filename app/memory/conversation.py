"""
In-memory conversation store.

Each conversation is identified by a session_id (string).
The store keeps the last MEMORY_WINDOW turns (user + assistant pairs).

In production you would replace the in-process dict with Redis or a DB,
but for a single-server FastAPI app this is perfectly fine.
"""

from collections import deque
from typing import List, Dict
from app.config import MEMORY_WINDOW


class ConversationMemory:
    """Sliding window of recent turns for one session."""

    def __init__(self, window: int = MEMORY_WINDOW):
        self._window = window
        # Each entry: {"role": "user"|"assistant", "content": str}
        self._turns: deque = deque(maxlen=window * 2)  # *2 for user+assistant

    def add_turn(self, user_message: str, assistant_message: str):
        self._turns.append({"role": "user",      "content": user_message})
        self._turns.append({"role": "assistant",  "content": assistant_message})

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._turns)

    def format_for_prompt(self) -> str:
        """Format history as a readable block for inclusion in prompts."""
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self._turns) == 0


# ── Session registry ──────────────────────────────────────────────────────────

_sessions: Dict[str, ConversationMemory] = {}


def get_or_create_session(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def delete_session(session_id: str):
    _sessions.pop(session_id, None)


def list_sessions() -> List[str]:
    return list(_sessions.keys())

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptContext:
    message: str = """
{user_context}
{context}
"""
    user_context: str = ""
    user_context_short: str = ""
    user: str = ""
    context: str = ""
    _history: list[dict] = field(default_factory=list)

    @property
    def system_message(self):
        return self.message.format(**vars(self))

    @property
    def history(self):
        h = [{"role": "system", "content": self.system_message}]
        h.extend(self._history)
        return h

    @property
    def conversation(self):
        return [
            entry
            for entry in self.history
            if entry.get("role") in ("user", "assistant")
        ]

    def push(self, *entry: dict):
        self._history.extend(entry)
        return self

    def push_user(self, content: str):
        self._history.append({"role": "user", "content": content})
        return self

    def push_assistant(self, content: str):
        self._history.append({"role": "assistant", "content": content})
        return self

    def get_item_from_history(self, item_id: str) -> Any | None:
        return next((entry for entry in self._history if entry.get("id") == item_id))

    def reset(self):
        self._history = []
        return self

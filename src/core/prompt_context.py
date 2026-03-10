from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptContext:
    message: str = ""
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

    def push(self, entry: Any):
        self._history.append(entry)
        return self

    def reset(self):
        self._history = []
        return self

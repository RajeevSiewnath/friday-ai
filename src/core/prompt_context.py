from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptContext:
    message: str = ""
    user_context: str = ""
    user_context_short: str = ""
    user: str = ""
    context: str = ""
    _history: list[Any] = field(default_factory=list)

    @property
    def system_message(self):
        return self.message.format(**vars(self))

    @property
    def history(self):
        pass

    def push(self, entry: Any):
        self._history.append(entry)

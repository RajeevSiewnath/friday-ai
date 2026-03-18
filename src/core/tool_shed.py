import json
from typing import Any, Callable, Self, TypedDict


class ToolDefinition(TypedDict):
    name: str
    callable: Callable
    definition: Any


class ToolShed:
    def __init__(self, *tools: ToolDefinition):
        self.tools: list[ToolDefinition] = tools

    def add(self, *tools: ToolDefinition) -> Self:
        self.tools.extend(*tools)
        return self

    def remove(self, *tools: ToolDefinition) -> Self:
        self.tools = [t for t in self.tools if t not in tools]
        return self

    def call(self, name: str, args: str) -> Any:
        tool = next([t for t in self.tools if t.name == name])
        if tool:
            return tool(**json.loads(args))
        else:
            raise f"tool not defined: '{name}'"

    @property
    def definitions(self) -> list[Any]:
        return [t["definition"] for t in self.tools]

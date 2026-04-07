from dataclasses import dataclass
import json
from typing import Any, Callable, Self
from ..utils.function_definition_creator import function_definition_creator


@dataclass
class ToolDefinition:
    name: str
    callable: Callable
    definition: Any


class ToolShed:
    def __init__(self, *tools: Any):
        self.tools: list[ToolDefinition] = []
        self.add(*tools)

    def add(self, *tools: Any) -> Self:
        tool_defs = []
        for t in tools:
            definition = function_definition_creator(t)
            tool_defs.append(
                ToolDefinition(
                    name=definition["name"],
                    definition=definition,
                    callable=t,
                )
            )
        self.tools.extend(tool_defs)
        return self

    def remove(self, *tools: ToolDefinition) -> Self:
        self.tools = [t for t in self.tools if t not in tools]
        return self

    def call(self, name: str, args: str) -> Any:

        tool = next((t for t in self.tools if t.name == name), None)
        if tool:
            return tool.callable(**json.loads(args))
        else:
            raise f"tool not defined: '{name}'"

    @property
    def definitions(self) -> list[Any]:
        return [t.definition for t in self.tools]

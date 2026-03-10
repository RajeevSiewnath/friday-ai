from typing import Any
from dotenv import load_dotenv
import json
from typing import Callable, Generator, Iterable, Self, TypedDict, Any
from enum import Enum
from openai import Omit
import warnings
from core.llm import LLM
from core.prompt_context import PromptContext

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
load_dotenv()


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Type(str, Enum):
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"


class HistoryEntry(TypedDict):
    role: Omit | Role
    content: Omit | str
    type: Omit | Type
    call_id: Omit | str
    output: str


class Tool(TypedDict):
    call_id: str
    name: str
    args: str


class ToolDefinition(TypedDict):
    name: str
    callable: Callable
    definition: Any


class ChatLoop:
    def __init__(
        self,
        llm: LLM,
        prompt_context: PromptContext,
        tools: Iterable[ToolDefinition] = [],
    ):
        self.llm = llm
        self.prompt_context = prompt_context
        self.register_tools(tools)

    def register_tools(self, tools: Iterable[Any]) -> Self:
        self.tools = tools
        return self

    def call_tool(self, callable: Tool) -> Any:
        name = callable["name"]
        args = json.loads(callable["args"])
        tool = next((tool for tool in self.tools if tool["name"] == name))
        if tool:
            value = str(tool["callable"](**args))
            return value
        else:
            raise f"illegal tool name '{name}'"

    def submit_message(self, message: str) -> Self:
        self.prompt_context.push({"role": Role.USER, "content": message})
        return self

    def get_item_from_history(self, item_id: str) -> Any | None:
        return next(
            (
                entry
                for entry in self.prompt_context.history
                if entry.get("id") == item_id
            )
        )

    def reset(self) -> Self:
        self.prompt_context.reset()
        return self

    def invoke(self) -> Generator[bool, None, None]:
        try:
            with self.llm.stream(
                tools=[tool["definition"] for tool in self.tools],
                input=self.prompt_context.history,
            ) as stream:
                for event in stream:
                    if event.type == "response.created":
                        pass
                    elif event.type == "response.in_progress":
                        pass
                    elif event.type == "response.completed":
                        yield True
                    elif event.type == "response.failed":
                        pass
                    elif event.type == "response.output_item.added":
                        self.prompt_context.push(event.item.model_dump())
                        yield False
                    elif event.type == "response.output_item.done":
                        pass
                    elif event.type == "response.content_part.added":
                        pass
                    elif event.type == "response.content_part.done":
                        pass
                    elif event.type == "response.output_text.delta":
                        item = self.get_item_from_history(event.item_id)
                        if item:
                            if isinstance(item["content"], list):
                                item["content"] = ""
                            item["content"] += event.delta
                        yield False
                    elif event.type == "response.output_text.done":
                        pass
                    elif event.type == "response.output_text.annotation_added":
                        pass
                    elif event.type == "response.text.done":
                        pass
                    elif event.type == "response.function_call_arguments.delta":
                        item = self.get_item_from_history(event.item_id)
                        if item:
                            item["arguments"] += event.delta
                    elif event.type == "response.function_call_arguments.done":
                        item = self.get_item_from_history(event.item_id)
                        if item:
                            result = self.call_tool(
                                {
                                    "call_id": item["call_id"],
                                    "name": item["name"],
                                    "args": item["arguments"],
                                }
                            )
                            self.prompt_context.push(
                                {
                                    "type": Type.FUNCTION_CALL_OUTPUT,
                                    "call_id": item["call_id"],
                                    "output": str(result),
                                }
                            )
                            for done in self.invoke():
                                yield done
                    elif event.type == "response.refusal.delta":
                        pass
                    elif event.type == "response.refusal.done":
                        pass
                    elif event.type == "response.file_search_call.in_progress":
                        pass
                    elif event.type == "response.file_search_call.searching":
                        pass
                    elif event.type == "response.file_search_call.completed":
                        pass
                    elif event.type == "error":
                        raise Exception(f"an unexpected error occurred")
                    else:
                        raise Exception(f"unsupported type {event.type}")
        except Exception as e:
            self.prompt_context.push(
                {"role": Role.ASSISTANT, "content": "Something went wrong, try again"}
            )
            print(e)
            yield True

import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
import json
from typing import Callable, Generator, Iterable, Self, TypedDict, Any
from enum import Enum
from colorama import Fore, Style
from openai import Omit, OpenAI
import os
import warnings

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
    client: OpenAI
    system_prompt: str
    history: list[HistoryEntry] = [{"role": Role.SYSTEM, "content": ""}]
    model: str
    tools: Iterable[ToolDefinition] = []
    log: set[str] = set()

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        system_prompt: str = "You are a helpful assistant.",
        initial_history: list[HistoryEntry] = None,
        tools: Iterable[ToolDefinition] = [],
    ):
        self.model = model
        self.set_system_prompt(system_prompt)
        self.register_tools(tools)
        if initial_history:
            self.set_initial_history(initial_history)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @property
    def conversation(self):
        return [
            entry
            for entry in self.history
            if entry.get("type") not in (Type.FUNCTION_CALL, Type.FUNCTION_CALL_OUTPUT)
        ]

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

    def set_system_prompt(self, system_prompt: str) -> Self:
        self.system_prompt = system_prompt
        return self

    def set_initial_history(self, initial_history: list[HistoryEntry]) -> Self:
        self.history = self.history[:1] + initial_history + self.history[1:]
        return self

    def submit_message(self, message: str) -> Self:
        self.history.append({"role": Role.USER, "content": message})
        return self

    def get_item_from_history(self, item_id: str) -> Any | None:
        return next((entry for entry in self.history if entry.get("id") == item_id))

    def reset(self) -> Self:
        self.history = [{"role": Role.SYSTEM, "content": ""}]
        return self

    def invoke(self) -> Generator[bool, None, None]:
        try:
            print(f"stream in")
            self.history[0]["content"] = self.system_prompt
            with self.client.responses.stream(
                model=self.model,
                tools=[tool["definition"] for tool in self.tools],
                input=self.history,
            ) as stream:
                for event in stream:
                    self.log.add(event.model_dump_json(indent=2))
                    print(event.type)
                    if event.type == "response.created":
                        pass
                    elif event.type == "response.in_progress":
                        pass
                    elif event.type == "response.completed":
                        yield True
                    elif event.type == "response.failed":
                        pass
                    elif event.type == "response.output_item.added":
                        print(
                            Fore.GREEN
                            + event.model_dump_json(indent=2)
                            + Style.RESET_ALL
                        )
                        self.history.append(event.item.model_dump())
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
                            self.history.append(
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
            print(f"stream out")
        except Exception as e:
            self.history.append(
                {"role": Role.ASSISTANT, "content": "Something went wrong, try again"}
            )
            print(e)
            yield True
            # if stream_id == None:
            #     print(Fore.BLUE + json.dumps(self.history, indent=2) + Style.RESET_ALL)
            #     for log in self.log:
            #         print(Fore.YELLOW + log + Style.RESET_ALL)
            #         print(
            #             Fore.GREEN
            #             + "---------------------------------------------------------"
            #             + Style.RESET_ALL
            #         )
            # raise e


# response.code_interpreter_call.* (various code/tool events)

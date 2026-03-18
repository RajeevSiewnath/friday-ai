from typing import Generator
import warnings
from core.llm import LLM
from core.prompt_context import PromptContext

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


class ChatLoop:
    def __init__(self, llm: LLM):
        self.llm = llm
        self._is_looping = False

    @property
    def is_looping(self):
        return self._is_looping

    def invoke(self, prompt_context: PromptContext) -> Generator[bool, None, None]:
        self._is_looping = True
        yield from self._invoke(prompt_context)
        self._is_looping = False

    def _invoke(self, prompt_context: PromptContext) -> Generator[bool, None, None]:
        try:
            with self.llm.stream(input=prompt_context.history) as stream:
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
                        prompt_context.push(event.item.model_dump())
                        yield False
                    elif event.type == "response.output_item.done":
                        pass
                    elif event.type == "response.content_part.added":
                        pass
                    elif event.type == "response.content_part.done":
                        pass
                    elif event.type == "response.output_text.delta":
                        item = prompt_context.get_item_from_history(event.item_id)
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
                        item = prompt_context.get_item_from_history(event.item_id)
                        if item:
                            item["arguments"] += event.delta
                    elif event.type == "response.function_call_arguments.done":
                        item = prompt_context.get_item_from_history(event.item_id)
                        if item:
                            result = self.llm.tool_shed.call(
                                item["name"], item["arguments"]
                            )
                            prompt_context.push(
                                {
                                    "type": "function_call_output",
                                    "call_id": item["call_id"],
                                    "output": str(result),
                                }
                            )
                            for done in self._invoke():
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
            prompt_context.push(
                {"role": "assistant", "content": "Something went wrong, try again"}
            )
            print(e)
            yield True

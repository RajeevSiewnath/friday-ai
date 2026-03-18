from typing import Any, Type, TypeVar
from pipelines.pipeline import Pipe
from models.query_context import QueryContext

T = TypeVar("T")


class InvokeLLM(Pipe[Any, str]):

    def __init__(self, response_format: Type[T] = str):
        super().__init__()
        self.response_format = response_format

    def run(self, input):
        return self.llm.invoke(
            input=self.prompt_context.history, response_format=self.response_format
        )

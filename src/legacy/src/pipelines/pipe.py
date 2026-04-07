from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB

I = TypeVar("I")
O = TypeVar("O", default=I)
Next = TypeVar("Next")


class Pipe(ABC, Generic[I, O]):
    def __init__(self):
        self.llm: Optional[LLM] = None
        self.prompt_context: Optional[PromptContext] = None
        self.vector_db: Optional[VectorDB] = None

    def set_environment(
        self, llm: LLM, prompt_context: PromptContext, vector_db: VectorDB
    ) -> None:
        self.llm = llm
        self.prompt_context = prompt_context
        self.vector_db = vector_db

    @abstractmethod
    def run(self, input: I) -> O: ...

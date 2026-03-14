from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Self
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB

class AbstractPipe[I, O = I](ABC):
    def __init__(self):
        self.llm: LLM = None
        self.prompt_context: PromptContext = None
        self.vector_db: VectorDB = None

    def _set_pipeline_environment(
        self, 
        llm: LLM, 
        prompt_context: PromptContext, 
        vector_db: VectorDB, 
    ):
        self.llm = llm
        self.prompt_context = prompt_context
        self.vector_db = vector_db

    @abstractmethod
    def pipe(self, input: I) -> O:
        pass


class AbstractPipeline[I, O = I](ABC):

    def __init__(
        self, 
        *pipes: AbstractPipe[I, O],
        llm: LLM, 
        prompt_context: PromptContext, 
        vector_db: VectorDB, 
    ):
        super().__init__()
        self.llm = llm
        self.prompt_context = prompt_context
        self.vector_db = vector_db
        self.pipes: list[AbstractPipe[I, O]] = [] 
        self.add(*pipes)

    def add(self, *pipes: AbstractPipe[I, O]) -> Self:
        for pipe in pipes:
            self.pipes.append(pipe)
        return self

    def remove(self, *pipes: AbstractPipe[I, O]) -> Self:
        for pipe in pipes:
            self.pipes.remove(pipe)
        return self
    
    def run(self, input: I) -> O:
        current = deepcopy(input)
        for pipe in self.pipes:
            pipe._set_pipeline_environment(
                llm = self.llm,
                prompt_context = self.prompt_context,
                vector_db = self.vector_db,
            )
            current = pipe.pipe(current)
        return current

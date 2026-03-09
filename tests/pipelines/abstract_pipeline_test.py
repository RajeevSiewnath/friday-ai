from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Self
from core.llm import LLM
from core.prompt_context import PromptContext

@dataclass
class PipeArg[I]:
    input: I
    prompt_context: PromptContext
    llm: LLM


class AbstractPipe[I, O = I](ABC):
    @abstractmethod
    def pipe(self, arg: PipeArg[I]) -> O:
        pass


class AbstractPipeline[I, O = I](ABC):
    pipes: list[AbstractPipe[I, O]] = []

    def __init__(self, *pipes: AbstractPipe[I, O]):
        super().__init__()
        self.add(*pipes)

    def add(self, *pipes: AbstractPipe[I, O]) -> Self:
        for pipe in pipes:
            self.pipes.append(pipe)
        return self

    def remove(self, *pipes: AbstractPipe[I, O]) -> Self:
        for pipe in pipes:
            self.pipes.remove(pipe)
        return self
    
    def run(self, arg: PipeArg[I]) -> O:
        current = deepcopy(arg.input)
        for pipe in self.pipes:
            arg.input = current
            current = pipe.pipe(arg)
        return current

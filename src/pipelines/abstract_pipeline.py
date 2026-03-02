from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Self, TypeVar


I = TypeVar("I")
O = TypeVar("O")


class AbstractPipe[I, O = I](ABC):
    @abstractmethod
    def pipe(self, input: I) -> O:
        pass


class AbstractPipeline[I, O = I](ABC):

    initial: I
    pipes: list[AbstractPipe[I, O]] = []

    def __init__(self, initial: I):
        super().__init__()
        self.initial = initial


    def add(self, pipe: AbstractPipe[I, O]) -> Self:
        self.pipes.append(pipe)
        return self

    def remove(self, pipe: AbstractPipe[I, O]) -> Self:
        self.pipes.remove(pipe)
        return self
    
    def run(self, input: I) -> O:
        output = deepcopy(input)
        for pipe in self.pipes:
            output = pipe.pipe(input)
        return output

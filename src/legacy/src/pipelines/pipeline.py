from copy import deepcopy
from typing import Generic, TypeVar
from pipelines.pipe import Pipe

I = TypeVar("I")
O = TypeVar("O", default=I)
Next = TypeVar("Next")


class Pipeline(Pipe[I, O], Generic[I, O]):
    def __init__(self, *pipes: Pipe[I, O]):
        super().__init__()
        self.pipes: list[Pipe] = list(pipes)

    def run(self, input: I) -> O:
        current: I | O = deepcopy(input)
        for pipe in self.pipes:
            pipe.set_environment(self.llm, self.prompt_context, self.vector_db)
            current = pipe.run(current)
        return current  # type: ignore

    def pipe(self, *next_pipes: Pipe[O, Next]) -> "Pipeline[I, Next]":
        pipeline = Pipeline(*self.pipes, *next_pipes)
        pipeline.set_environment(self.llm, self.prompt_context, self.vector_db)
        return pipeline

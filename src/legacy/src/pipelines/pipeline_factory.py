from typing import ParamSpec, Type, TypeVar
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from pipelines.pipe import Pipe
from pipelines.pipeline import Pipeline

I = TypeVar("I")
O = TypeVar("O")
P = TypeVar("P", bound=Pipeline)
PArgs = ParamSpec("PArgs")  # parameters of the pipeline constructor


class PipelineFactory:
    def __init__(
        self, llm: LLM, prompt_context: PromptContext, vector_db: VectorDB
    ) -> None:
        self.llm = llm
        self.prompt_context = prompt_context
        self.vector_db = vector_db

    def make(
        self,
        *pipes: Pipe[I, O],
        pipeline_cls: Type[P] = Pipeline[I, O],
        **kwargs: PArgs.kwargs,
    ) -> P:
        pipeline = pipeline_cls(*pipes, **kwargs)
        pipeline.set_environment(self.llm, self.prompt_context, self.vector_db)
        return pipeline

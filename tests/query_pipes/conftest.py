import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from pipelines.abstract_pipeline import PipeArg
from pipelines.query_pipeline import QueryContext, QueryPipeline


@pytest.fixture
def query_pipe_arg(llm: LLM, prompt_context: PromptContext, vector_db: VectorDB):
    return PipeArg[QueryContext](
        input=QueryContext("Who is Rajeev?"),
        llm=llm,
        prompt_context=prompt_context,
        vector_db=vector_db,
    )


@pytest.fixture(scope="function")
def query_pipeline():
    return QueryPipeline()

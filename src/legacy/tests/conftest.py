import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from pipelines.pipeline_factory import PipelineFactory


@pytest.fixture
def prompt_context() -> PromptContext:
    return PromptContext(
        message="""
{user_context}

Context:
{context}
""",
        user="Rajeev Siewnath",
        user_context="You're an agent for Rajeev Siewnath",
        user_context_short="Rajeev Siewnath's agent",
        context="",
    )


@pytest.fixture
def llm():
    return LLM()


@pytest.fixture
def vector_db(llm):
    return VectorDB(llm, "cv-rajeev-siewnath")


@pytest.fixture
def pipeline_factory(prompt_context: PromptContext, llm: LLM, vector_db: VectorDB):
    return PipelineFactory(llm, prompt_context, vector_db)

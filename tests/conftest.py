import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB


@pytest.fixture
def prompt_context() -> PromptContext:
    return PromptContext(
        message="""
{user_context}

Context:
{context}
"""
    )


@pytest.fixture
def llm():
    return LLM()


@pytest.fixture
def vector_db(llm):
    return VectorDB(llm, "cv-rajeev-siewnath")

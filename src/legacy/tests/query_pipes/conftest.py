import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from models.query_context import QueryContext


@pytest.fixture
def query_pipe_arg():
    return QueryContext("Who is Rajeev?")

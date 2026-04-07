import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from document_loader_pipes.document_loader import DocumentLoader
from models.document import DocumentCollection


@pytest.fixture
def document_loader_pipe():
    return DocumentLoader("data", 2)


@pytest.fixture
def document_loader_pipe_full():
    return DocumentLoader("data")


@pytest.fixture
def document_loader_pipe_arg():
    return DocumentCollection()

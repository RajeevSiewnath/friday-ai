import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from document_loader_pipes.document_loader import DocumentLoader
from pipelines.abstract_pipeline import PipeArg
from pipelines.document_loader_pipeline import (
    DocumentCollection,
    DocumentLoaderPipeline,
)


@pytest.fixture
def document_loader_pipe():
    return DocumentLoader("data", 2)


@pytest.fixture
def document_loader_pipe_full():
    return DocumentLoader("data")


@pytest.fixture
def document_loader_pipe_arg(
    llm: LLM, prompt_context: PromptContext, vector_db: VectorDB
):
    return PipeArg[DocumentCollection](
        input=DocumentCollection(),
        llm=llm,
        prompt_context=prompt_context,
        vector_db=vector_db,
    )


@pytest.fixture(scope="function")
def document_loader_pipeline():
    return DocumentLoaderPipeline()

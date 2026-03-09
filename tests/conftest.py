import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from document_loader_pipes.document_loader import DocumentLoader
from pipelines.abstract_pipeline import PipeArg
from pipelines.document_loader_pipeline import (
    DocumentCollection,
    DocumentLoaderPipeline,
)
from pipelines.evaluation_pipeline import EvaluationPipeline, EvaluationScore
from pipelines.query_pipeline import QueryContext, QueryPipeline


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
def llm() -> LLM:
    return LLM()


@pytest.fixture
def document_loader_pipe_arg(llm: LLM, prompt_context: PromptContext):
    return PipeArg[DocumentCollection](
        input=DocumentCollection(), llm=llm, prompt_context=prompt_context
    )


@pytest.fixture
def document_loader_pipe():
    return DocumentLoader("data", 2)


@pytest.fixture
def document_loader_pipe_full():
    return DocumentLoader("data")


@pytest.fixture
def document_loader_pipeline(document_loader_pipe: DocumentLoader):
    return DocumentLoaderPipeline(document_loader_pipe)


@pytest.fixture
def evaluation_pipe_arg(llm: LLM, prompt_context: PromptContext):
    return PipeArg[EvaluationScore](
        input=EvaluationScore(), llm=llm, prompt_context=prompt_context
    )


@pytest.fixture
def evaluation_pipeline(document_loader_pipe_arg: PipeArg[EvaluationScore]):
    return EvaluationPipeline(document_loader_pipe_arg)


@pytest.fixture
def query_pipe_arg(llm: LLM, prompt_context: PromptContext):
    return PipeArg[QueryContext](
        input=QueryContext(), llm=llm, prompt_context=prompt_context
    )


@pytest.fixture
def query_pipeline(document_loader_pipe_arg: PipeArg[QueryContext]):
    return QueryPipeline(document_loader_pipe_arg)

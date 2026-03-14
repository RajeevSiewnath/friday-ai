from core.prompt_context import PromptContext
from models.query_context import QueryContext
from pipelines.query_pipeline import QueryPipeline
from query_pipes.rag_context_injector import RagContextInjector
from query_pipes.rag_context_retriever import RagContextRetriever


def test_rag_context_injector(
    query_pipeline: QueryPipeline,
    query_pipe_arg: QueryContext,
    prompt_context: PromptContext,
):
    query_pipeline.add(
        RagContextRetriever(),
        RagContextInjector(),
    ).run(query_pipe_arg)
    assert len(prompt_context.context) > 0

from core.prompt_context import PromptContext
from models.query_context import QueryContext
from query_pipes.rag_context_injector import RagContextInjector
from query_pipes.rag_context_retriever import RagContextRetriever
from pipelines.pipeline_factory import PipelineFactory


def test_rag_context_injector(
    pipeline_factory: PipelineFactory,
    query_pipe_arg: QueryContext,
    prompt_context: PromptContext,
):
    pipeline_factory.make(
        RagContextRetriever(),
        RagContextInjector(),
    ).run(query_pipe_arg)
    assert len(prompt_context.context) > 0

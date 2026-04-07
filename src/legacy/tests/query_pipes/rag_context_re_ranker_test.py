from models.query_context import QueryContext
from query_pipes.rag_context_re_ranker import RagContextReRanker
from query_pipes.rag_context_retriever import RagContextRetriever
from pipelines.pipeline_factory import PipelineFactory


def test_rag_context_re_ranker(
    pipeline_factory: PipelineFactory,
    query_pipe_arg: QueryContext,
):
    query_context = pipeline_factory.make(
        RagContextRetriever(),
        RagContextReRanker(),
    ).run(query_pipe_arg)
    assert len(query_context.context.contexts) > 0

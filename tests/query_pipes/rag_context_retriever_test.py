from models.query_context import QueryContext
from pipelines.abstract_pipeline import PipeArg
from pipelines.query_pipeline import QueryPipeline
from query_pipes.rag_context_retriever import RagContextRetriever


def test_rag_context_retriever(
    query_pipeline: QueryPipeline,
    query_pipe_arg: PipeArg[QueryContext],
):
    query_context: QueryContext = query_pipeline.add(RagContextRetriever()).run(
        query_pipe_arg
    )
    assert len(query_context.context.contexts) > 0

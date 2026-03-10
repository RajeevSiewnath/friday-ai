from models.query_context import QueryContext
from pipelines.abstract_pipeline import PipeArg
from pipelines.query_pipeline import QueryPipeline
from query_pipes.query_rewriter import QueryRewriter


def test_query_rewriter(
    query_pipeline: QueryPipeline,
    query_pipe_arg: PipeArg[QueryContext],
):
    query_context: QueryContext = query_pipeline.add(QueryRewriter()).run(
        query_pipe_arg
    )
    assert len(query_context.question_history) == 2

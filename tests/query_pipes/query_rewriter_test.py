from models.query_context import QueryContext
from query_pipes.query_rewriter import QueryRewriter
from pipelines.pipeline_factory import PipelineFactory


def test_query_rewriter(
    pipeline_factory: PipelineFactory,
    query_pipe_arg: QueryContext,
):
    query_context = pipeline_factory.make(QueryRewriter()).run(query_pipe_arg)
    assert len(query_context.question_history) == 2

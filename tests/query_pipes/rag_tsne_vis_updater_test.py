import pytest
from models.query_context import QueryContext
from pipelines.query_pipeline import QueryPipeline
from query_pipes.rag_context_retriever import RagContextRetriever


@pytest.mark.skip("Not implemented yet")
def test_tsne_vis_updater(
    query_pipeline: QueryPipeline,
    query_pipe_arg: QueryContext,
):
    pass

from document_loader_pipes.document_loader import DocumentLoader
from models.document import DocumentCollection
from pipelines.pipeline_factory import PipelineFactory


def test_document_loader(
    pipeline_factory: PipelineFactory,
    document_loader_pipe_full: DocumentLoader,
    document_loader_pipe_arg: DocumentCollection,
):
    document_collection = pipeline_factory.make(
        document_loader_pipe_full,
    ).run(document_loader_pipe_arg)
    assert len(document_collection.documents) > 0

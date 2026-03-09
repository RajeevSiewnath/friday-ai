from document_loader_pipes.document_loader import DocumentLoader
from pipelines.abstract_pipeline import PipeArg
from pipelines.document_loader_pipeline import (
    DocumentCollection,
    DocumentLoaderPipeline,
)


def test_document_loader(
    document_loader_pipeline: DocumentLoaderPipeline,
    document_loader_pipe_full: DocumentLoader,
    document_loader_pipe_arg: PipeArg[DocumentCollection],
):
    document_collection: DocumentCollection = document_loader_pipeline.add(
        document_loader_pipe_full,
    ).run(document_loader_pipe_arg)
    assert len(document_collection.documents) > 0

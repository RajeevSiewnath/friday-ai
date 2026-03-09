from document_loader_pipes.chunk_splitter import ChunkSplitter
from document_loader_pipes.document_loader import DocumentLoader
from pipelines.abstract_pipeline import PipeArg
from pipelines.document_loader_pipeline import (
    DocumentCollection,
    DocumentLoaderPipeline,
)


def test_chunk_splitter(
    document_loader_pipeline: DocumentLoaderPipeline,
    document_loader_pipe: DocumentLoader,
    document_loader_pipe_arg: PipeArg[DocumentCollection],
):
    document_collection: DocumentCollection = document_loader_pipeline.add(
        document_loader_pipe,
    ).run(document_loader_pipe_arg)
    assert len(document_collection.documents) > 0
    splitted_document_collection: DocumentCollection = document_loader_pipeline.add(
        document_loader_pipe, ChunkSplitter()
    ).run(document_loader_pipe_arg)
    assert len(splitted_document_collection.documents) > 0
    assert len(splitted_document_collection.documents) > len(
        document_collection.documents
    )

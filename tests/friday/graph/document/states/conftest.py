import pytest
from friday.core.document import Document
from friday.graph.document.states.document_state import DocumentState


@pytest.fixture
def sample_document():
    return Document(
        id="test_doc",
        path="test.json",
        type="test_type",
        content="Test document content",
        metadata={"test": True, "version": 1},
    )


@pytest.fixture
def multiple_documents():
    return [
        Document(
            id="doc_1",
            path="path_1.json",
            type="type_1",
            content="Content 1",
            metadata={"order": 1},
        ),
        Document(
            id="doc_2",
            path="path_2.json",
            type="type_2",
            content="Content 2",
            metadata={"order": 2},
        ),
        Document(
            id="doc_3",
            path="path_3.json",
            type="type_3",
            content="Content 3",
            metadata={"order": 3},
        ),
    ]


@pytest.fixture
def document_state_with_single(sample_document):
    return DocumentState(documents=[sample_document])

import pytest
from friday.core.document import Document
from friday.graph.document.reducers.document_reducer import DocumentReducerClearAction


@pytest.fixture
def sample_document():
    return Document(
        id="doc1",
        path="test.json",
        type="test",
        content="Test content",
        metadata={"test": True},
    )


@pytest.fixture
def document_list():
    return [
        Document(
            id="doc1",
            path="path1.json",
            type="type1",
            content="Content 1",
            metadata={"id": 1},
        ),
        Document(
            id="doc2",
            path="path2.json",
            type="type2",
            content="Content 2",
            metadata={"id": 2},
        ),
    ]


@pytest.fixture
def clear_action(document_list):
    return DocumentReducerClearAction(document_list)

import pytest
from typing import get_args, get_origin
from friday.core.document import Document
from friday.graph.document.states.document_state import DocumentState


class TestDocumentState:
    def test_document_state_creation(self, document_state: DocumentState):
        assert "documents" in document_state
        assert isinstance(document_state["documents"], list)

    def test_document_state_has_documents_key(self, document_state: DocumentState):
        assert "documents" in document_state

    def test_document_state_documents_is_list(self, document_state: DocumentState):
        assert isinstance(document_state["documents"], list)

    def test_document_state_contains_document_objects(self, document_state: DocumentState):
        assert all(isinstance(doc, Document) for doc in document_state["documents"])

    def test_document_state_empty(self, empty_document_state: DocumentState):
        assert empty_document_state["documents"] == []

    def test_document_state_annotations(self):
        assert hasattr(DocumentState, "__annotations__")
        assert "documents" in DocumentState.__annotations__

    def test_document_state_multiple_documents(self, document_state: DocumentState):
        assert len(document_state["documents"]) > 0
        for doc in document_state["documents"]:
            assert hasattr(doc, "id")
            assert hasattr(doc, "path")
            assert hasattr(doc, "type")
            assert hasattr(doc, "content")
            assert hasattr(doc, "metadata")

    def test_document_state_document_attributes(self, document_state: DocumentState):
        doc = document_state["documents"][0]
        assert isinstance(doc.id, str)
        assert isinstance(doc.content, str)
        assert isinstance(doc.metadata, dict)

    def test_document_state_can_be_modified(self, document_state: DocumentState):
        original_count = len(document_state["documents"])
        new_doc = Document(
            id="new_doc",
            content="New content",
            path="new.json",
            type="new",
            metadata={"new": True},
        )
        document_state["documents"].append(new_doc)
        assert len(document_state["documents"]) == original_count + 1

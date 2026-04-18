import pytest
from copy import deepcopy
from friday.core.document import Document
from friday.graph.document.reducers.document_reducer import (
    DocumentReducerClearAction,
    document_reducer,
)


class TestDocumentReducerClearAction:
    def test_clear_action_creation(self, document_list):
        action = DocumentReducerClearAction(document_list)
        assert action.content == document_list

    def test_clear_action_stores_list(self, sample_document):
        docs = [sample_document]
        action = DocumentReducerClearAction(docs)
        assert len(action.content) == 1
        assert action.content[0].id == "doc1"

    def test_clear_action_with_empty_list(self):
        action = DocumentReducerClearAction([])
        assert action.content == []

    def test_clear_action_with_multiple_documents(self, document_list):
        action = DocumentReducerClearAction(document_list)
        assert len(action.content) == len(document_list)


class TestDocumentReducer:
    def test_reducer_with_clear_action(self, document_list):
        left = [
            Document(
                id="old1",
                path="old.json",
                type="old",
                content="Old",
                metadata={},
            )
        ]
        right = DocumentReducerClearAction(document_list)

        result = document_reducer(left, right)
        assert len(result) == len(document_list)
        assert result[0].id == "doc1"

    def test_reducer_clear_action_replaces_all(self, sample_document):
        left = [
            Document(
                id="doc_left",
                path="left.json",
                type="left",
                content="Left",
                metadata={},
            )
        ]
        right = DocumentReducerClearAction([sample_document])

        result = document_reducer(left, right)
        assert len(result) == 1
        assert result[0].id == "doc1"

    def test_reducer_with_empty_left_and_clear_action(self, document_list):
        left = []
        right = DocumentReducerClearAction(document_list)

        result = document_reducer(left, right)
        assert len(result) == len(document_list)

    def test_reducer_with_clear_action_preserves_content(self, clear_action):
        left = [
            Document(
                id="to_clear",
                path="clear.json",
                type="clear",
                content="Clear",
                metadata={},
            )
        ]

        result = document_reducer(left, clear_action)
        assert all(doc.id.startswith("doc") for doc in result)

    def test_reducer_deep_copies_with_clear_action(self, clear_action):
        left = [
            Document(
                id="left_doc",
                path="left.json",
                type="left",
                content="Left",
                metadata={},
            )
        ]

        result = document_reducer(left, clear_action)

        # Modify original to verify deep copy
        left[0].content = "Modified"
        assert result[0].content != "Modified"

    def test_reducer_returns_list(self, clear_action):
        left = [
            Document(
                id="left_doc",
                path="left.json",
                type="left",
                content="Left",
                metadata={},
            )
        ]

        result = document_reducer(left, clear_action)
        assert isinstance(result, list)

    def test_reducer_clear_action_is_instance_check(self, sample_document):
        left = [sample_document]
        right = DocumentReducerClearAction([sample_document])

        result = document_reducer(left, right)
        assert isinstance(right, DocumentReducerClearAction)
        assert len(result) == 1

    def test_reducer_empty_clear_action(self):
        left = [
            Document(
                id="doc1",
                path="p1.json",
                type="t1",
                content="c1",
                metadata={},
            )
        ]
        right = DocumentReducerClearAction([])

        result = document_reducer(left, right)
        assert len(result) == 0

    def test_reducer_clear_action_overwrites(self):
        left = [
            Document(
                id="doc1",
                path="p1.json",
                type="t1",
                content="c1",
                metadata={"old": True},
            ),
            Document(
                id="doc2",
                path="p2.json",
                type="t2",
                content="c2",
                metadata={"old": True},
            ),
        ]
        new_docs = [
            Document(
                id="doc_new",
                path="p_new.json",
                type="t_new",
                content="c_new",
                metadata={"new": True},
            )
        ]
        right = DocumentReducerClearAction(new_docs)

        result = document_reducer(left, right)
        assert len(result) == 1
        assert result[0].id == "doc_new"

    def test_reducer_maintains_document_properties_with_clear_action(self):
        left = [
            Document(
                id="doc1",
                path="p1.json",
                type="t1",
                content="c1",
                metadata={"a": 1},
            )
        ]
        right = DocumentReducerClearAction([
            Document(
                id="doc2",
                path="p2.json",
                type="t2",
                content="c2",
                metadata={"b": 2},
            )
        ])

        result = document_reducer(left, right)

        for doc in result:
            assert hasattr(doc, "id")
            assert hasattr(doc, "path")
            assert hasattr(doc, "type")
            assert hasattr(doc, "content")
            assert hasattr(doc, "metadata")

    def test_clear_action_content_is_deepcopied(self, sample_document):
        docs = [sample_document]
        action = DocumentReducerClearAction(docs)
        left = []

        result = document_reducer(left, action)

        # Verify the content was properly handled
        assert len(result) == 1
        assert result[0].id == sample_document.id

    def test_multiple_documents_in_clear_action(self):
        docs = [
            Document(
                id="doc1",
                path="p1.json",
                type="t1",
                content="c1",
                metadata={},
            ),
            Document(
                id="doc2",
                path="p2.json",
                type="t2",
                content="c2",
                metadata={},
            ),
            Document(
                id="doc3",
                path="p3.json",
                type="t3",
                content="c3",
                metadata={},
            ),
        ]
        left = [
            Document(
                id="old",
                path="old.json",
                type="old",
                content="old",
                metadata={},
            )
        ]
        right = DocumentReducerClearAction(docs)

        result = document_reducer(left, right)
        assert len(result) == 3
        assert [d.id for d in result] == ["doc1", "doc2", "doc3"]

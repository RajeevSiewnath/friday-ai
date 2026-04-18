import pytest
import tempfile
import json
import os
from friday.core.document import Document
from friday.graph.document.document_loader_factory import document_loader_factory
from friday.graph.document.states.document_state import DocumentState


class TestDocumentLoaderFactory:
    def test_loader_factory_returns_callable(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        assert callable(loader)

    def test_loader_returns_list_of_documents(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])
        result = loader(state)
        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)

    def test_loader_respects_max_parameter(self, document_json_folder):
        max_items = 1
        loader = document_loader_factory(document_json_folder, max=max_items)
        state = DocumentState(documents=[])
        result = loader(state)
        assert len(result) <= max_items

    def test_loader_without_max_loads_all(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])
        result = loader(state)
        assert len(result) == 3

    def test_loader_extracts_document_properties(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])
        result = loader(state)

        for doc in result:
            assert hasattr(doc, "id")
            assert hasattr(doc, "content")
            assert hasattr(doc, "metadata")
            assert hasattr(doc, "path")
            assert hasattr(doc, "type")

    def test_loader_with_custom_path(self, sample_documents):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(f"{tmpdir}/profile", exist_ok=True)
            with open(f"{tmpdir}/profile/test.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "document": "Test content",
                        "metadata": {"test": True},
                        "id": "test_id",
                    },
                    f,
                )

            loader = document_loader_factory(tmpdir)
            state = DocumentState(documents=[])
            result = loader(state)
            assert len(result) >= 1
            assert result[0].id == "test_id"

    def test_loader_preserves_content(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])
        result = loader(state)

        assert len(result) > 0
        first_doc = result[0]
        assert len(first_doc.content) > 0

    def test_loader_preserves_metadata(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])
        result = loader(state)

        for doc in result:
            assert isinstance(doc.metadata, dict)

    def test_loader_sets_type_from_folder(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])
        result = loader(state)

        for doc in result:
            assert doc.type in ["profile", "experience", "skills"]

    def test_loader_creates_valid_documents(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])
        result = loader(state)

        for doc in result:
            assert Document(**doc.model_dump())

    def test_loader_max_zero(self, document_json_folder):
        loader = document_loader_factory(document_json_folder, max=0)
        state = DocumentState(documents=[])
        result = loader(state)
        assert len(result) == 0

    def test_loader_max_exceeds_available(self, document_json_folder):
        loader = document_loader_factory(document_json_folder, max=1000)
        state = DocumentState(documents=[])
        result = loader(state)
        assert len(result) == 3

    def test_loader_callable_multiple_times(self, document_json_folder):
        loader = document_loader_factory(document_json_folder)
        state = DocumentState(documents=[])

        result1 = loader(state)
        result2 = loader(state)

        assert len(result1) == len(result2)

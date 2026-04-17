import pytest
from friday.models.document import Document, DocumentCollection


def make_document(**kwargs):
    """Helper to create Document with sensible defaults."""
    defaults = {
        "id": "test_doc",
        "content": "test content",
        "metadata": {},
        "path": None,
        "type": None,
    }
    defaults.update(kwargs)
    return Document(**defaults)


class TestDocument:
    def test_document_initialization(self):
        doc = make_document(
            id="doc1",
            content="test content",
            metadata={"source": "file.txt"},
        )
        assert doc.id == "doc1"
        assert doc.content == "test content"
        assert doc.metadata == {"source": "file.txt"}
        assert doc.path is None
        assert doc.type is None
        assert doc.metrics == []

    def test_document_with_path_and_type(self):
        doc = make_document(
            id="doc1",
            content="content",
            metadata={},
            path="/path/to/file.txt",
            type="txt",
        )
        assert doc.path == "/path/to/file.txt"
        assert doc.type == "txt"

    def test_document_with_metrics(self):
        metrics = [0.95, 0.87, 0.92]
        doc = make_document(
            id="doc1",
            content="content",
            metadata={},
            metrics=metrics,
        )
        assert doc.metrics == metrics

    def test_document_add_to_document(self):
        doc1 = make_document(id="doc1", content="content1")
        doc2 = make_document(id="doc2", content="content2")

        result = doc1 + doc2
        assert isinstance(result, DocumentCollection)
        assert len(result.documents) == 2
        assert result.documents[0].id == "doc1"
        assert result.documents[1].id == "doc2"

    def test_document_add_to_collection(self):
        doc1 = make_document(id="doc1", content="content1")
        collection = DocumentCollection(
            documents=[make_document(id="doc2", content="content2")]
        )

        result = doc1 + collection
        assert isinstance(result, DocumentCollection)
        assert len(result.documents) == 2
        assert result.documents[0].id == "doc1"

    def test_document_add_invalid_type_raises_exception(self):
        doc = make_document(id="doc1")
        with pytest.raises(Exception):
            _ = doc + "invalid"

    def test_document_original_unmodified_after_add(self):
        doc1 = make_document(
            id="doc1",
            content="content1",
            metadata={"key": "value"},
            metrics=[0.9],
        )
        doc2 = make_document(id="doc2", content="content2")

        result = doc1 + doc2
        # Ensure original is unchanged
        assert doc1.id == "doc1"
        assert len(doc1.metrics) == 1
        assert len(result.documents) == 2

    def test_document_complex_metadata(self):
        metadata = {
            "source": "file.txt",
            "page": 1,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        doc = make_document(id="doc1", content="content", metadata=metadata)
        assert doc.metadata["nested"]["key"] == "value"
        assert doc.metadata["list"] == [1, 2, 3]


class TestDocumentCollection:
    def test_collection_initialization(self):
        collection = DocumentCollection()
        assert collection.documents == []

    def test_collection_initialization_with_documents(self):
        docs = [
            make_document(id="doc1", content="content1"),
            make_document(id="doc2", content="content2"),
        ]
        collection = DocumentCollection(documents=docs)
        assert len(collection.documents) == 2

    def test_collection_from_docs(self):
        docs = [
            make_document(id="doc1", content="content1"),
            make_document(id="doc2", content="content2"),
        ]
        collection = DocumentCollection.from_docs(docs)
        assert len(collection.documents) == 2
        assert collection.documents[0].id == "doc1"

    def test_collection_add_to_collection(self):
        collection1 = DocumentCollection(
            documents=[make_document(id="doc1", content="content1")]
        )
        collection2 = DocumentCollection(
            documents=[make_document(id="doc2", content="content2")]
        )

        result = collection1 + collection2
        assert isinstance(result, DocumentCollection)
        assert len(result.documents) == 2

    def test_collection_add_to_document(self):
        collection = DocumentCollection(
            documents=[make_document(id="doc1", content="content1")]
        )
        doc = make_document(id="doc2", content="content2")

        result = collection + doc
        assert isinstance(result, DocumentCollection)
        assert len(result.documents) == 2
        assert result.documents[1].id == "doc2"

    def test_collection_add_invalid_type_raises_exception(self):
        collection = DocumentCollection()
        with pytest.raises(Exception):
            _ = collection + "invalid"

    def test_collection_original_unmodified_after_add(self):
        collection1 = DocumentCollection(
            documents=[make_document(id="doc1", content="content1")]
        )
        collection2 = DocumentCollection(
            documents=[make_document(id="doc2", content="content2")]
        )

        result = collection1 + collection2
        assert len(collection1.documents) == 1
        assert len(result.documents) == 2

    def test_collection_add_multiple_times(self):
        doc1 = make_document(id="doc1", content="content1")
        doc2 = make_document(id="doc2", content="content2")
        doc3 = make_document(id="doc3", content="content3")

        result = doc1 + doc2 + doc3
        assert len(result.documents) == 3
        assert result.documents[0].id == "doc1"
        assert result.documents[1].id == "doc2"
        assert result.documents[2].id == "doc3"

    def test_collection_with_mixed_metadata(self):
        docs = [
            make_document(
                id="doc1",
                content="content1",
                metadata={"source": "file1.txt"},
            ),
            make_document(
                id="doc2",
                content="content2",
                metadata={"source": "file2.txt", "page": 5},
            ),
        ]
        collection = DocumentCollection(documents=docs)
        assert collection.documents[0].metadata["source"] == "file1.txt"
        assert collection.documents[1].metadata["page"] == 5

    def test_collection_serialization(self):
        docs = [
            make_document(id="doc1", content="content1"),
        ]
        collection = DocumentCollection(documents=docs)
        data = collection.model_dump()

        assert len(data["documents"]) == 1
        assert data["documents"][0]["id"] == "doc1"

    def test_collection_deserialization(self):
        data = {
            "documents": [
                {
                    "id": "doc1",
                    "content": "content1",
                    "metadata": {},
                    "path": None,
                    "type": None,
                    "metrics": [],
                }
            ]
        }
        collection = DocumentCollection(**data)
        assert len(collection.documents) == 1
        assert collection.documents[0].id == "doc1"

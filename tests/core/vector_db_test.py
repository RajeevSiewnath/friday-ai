from core.vector_db import VectorDB
from models.document import Document, DocumentCollection


def test_vector_db_init(llm):
    """Test that VectorDB initializes with a chroma collection."""
    with VectorDB(llm, name="test-collection") as db:
        assert db.llm is llm
        assert db.collection is not None
        assert db.collection.name == "test-collection"


def test_vector_db_init_with_default_name(llm):
    """Test that VectorDB uses default name if not provided."""
    with VectorDB(llm, name="test-default") as db:
        assert db.collection.name == "test-default"


def test_populate_adds_documents(llm):
    """Test that populate() adds documents to the collection."""
    with VectorDB(llm, name="test-populate") as db:
        docs = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Python is a programming language",
                    path="python.txt",
                    type="text",
                    metadata={"section": "intro"},
                ),
                Document(
                    id="doc2",
                    content="JavaScript is used for web development",
                    path="javascript.txt",
                    type="text",
                    metadata={"section": "intro"},
                ),
            ]
        )

        db.populate(docs)

        # Verify documents are in the collection
        collection_data = db.collection.get()
        assert "doc1" in collection_data["ids"]
        assert "doc2" in collection_data["ids"]


def test_populate_returns_self(llm):
    """Test that populate() returns self for method chaining."""
    with VectorDB(llm, name="test-populate-chain") as db:
        docs = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Test content",
                    path="test.txt",
                    type="text",
                    metadata={},
                )
            ]
        )

        result = db.populate(docs)
        assert result is db


def test_populate_stores_metadata(llm):
    """Test that populate() stores document metadata correctly."""
    with VectorDB(llm, name="test-metadata") as db:
        docs = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Test content",
                    path="test.txt",
                    type="text",
                    metadata={"author": "Rajeev", "section": "work"},
                )
            ]
        )

        db.populate(docs)

        collection_data = db.collection.get(ids=["doc1"])
        metadata = collection_data["metadatas"][0]

        assert metadata["doc_type"] == "text"
        assert metadata["file_path"] == "test.txt"
        assert metadata["author"] == "Rajeev"
        assert metadata["section"] == "work"


def test_populate_with_force_flag_allows_repopulation(llm):
    """Test that populate() with force=True allows repopulation even if collection has documents."""
    with VectorDB(llm, name="test-force") as db:
        docs1 = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Original content",
                    path="test.txt",
                    type="text",
                    metadata={"version": "1"},
                )
            ]
        )

        db.populate(docs1)
        collection_data = db.collection.get()
        assert len(collection_data["ids"]) == 1

        docs2 = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Updated content",
                    path="test.txt",
                    type="text",
                    metadata={"version": "2"},
                )
            ]
        )

        # With force=True but clear=False, doc1 gets updated but not deleted
        db.populate(docs2, force=True, clear=False)

        collection_data = db.collection.get(ids=["doc1"])
        metadata = collection_data["metadatas"][0]
        assert metadata["version"] == "2"
        assert len(collection_data["ids"]) == 1


def test_populate_with_clear_flag_deletes_existing_documents(llm):
    """Test that populate() with force=True and clear=True deletes all existing documents."""
    with VectorDB(llm, name="test-clear") as db:
        docs1 = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Original doc 1",
                    path="doc1.txt",
                    type="text",
                    metadata={},
                ),
                Document(
                    id="doc2",
                    content="Original doc 2",
                    path="doc2.txt",
                    type="text",
                    metadata={},
                ),
            ]
        )

        db.populate(docs1)

        collection_data = db.collection.get()
        assert len(collection_data["ids"]) == 2

        docs2 = DocumentCollection(
            documents=[
                Document(
                    id="doc3",
                    content="New doc 3",
                    path="doc3.txt",
                    type="text",
                    metadata={},
                )
            ]
        )

        db.populate(docs2, force=True, clear=True)

        collection_data = db.collection.get()
        # Should only have the new document
        assert len(collection_data["ids"]) == 1
        assert "doc3" in collection_data["ids"]
        assert "doc1" not in collection_data["ids"]
        assert "doc2" not in collection_data["ids"]


def test_populate_with_clear_false_preserves_documents(llm):
    """Test that populate() with force=True and clear=False preserves existing documents."""
    with VectorDB(llm, name="test-no-clear") as db:
        docs1 = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Original doc",
                    path="doc1.txt",
                    type="text",
                    metadata={},
                )
            ]
        )

        db.populate(docs1)

        docs2 = DocumentCollection(
            documents=[
                Document(
                    id="doc2",
                    content="New doc",
                    path="doc2.txt",
                    type="text",
                    metadata={},
                )
            ]
        )

        db.populate(docs2, force=True, clear=False)

        collection_data = db.collection.get()
        # Should have both documents
        assert len(collection_data["ids"]) == 2
        assert "doc1" in collection_data["ids"]
        assert "doc2" in collection_data["ids"]


def test_populate_skips_if_collection_not_empty_and_force_false(llm):
    """Test that populate() skips if collection has documents and force=False."""
    with VectorDB(llm, name="test-skip-populate") as db:
        docs1 = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Original content",
                    path="doc1.txt",
                    type="text",
                    metadata={"version": "1"},
                )
            ]
        )

        db.populate(docs1)

        docs2 = DocumentCollection(
            documents=[
                Document(
                    id="doc2",
                    content="Second doc",
                    path="doc2.txt",
                    type="text",
                    metadata={"version": "2"},
                )
            ]
        )

        # This should not add doc2 since force=False and collection is not empty
        db.populate(docs2, force=False)

        collection_data = db.collection.get()
        # Should still only have doc1
        assert len(collection_data["ids"]) == 1
        assert "doc1" in collection_data["ids"]
        assert "doc2" not in collection_data["ids"]


def test_query_returns_document_collection(llm):
    """Test that query() returns a DocumentCollection."""
    with VectorDB(llm, name="test-query") as db:
        docs = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Python is a programming language used for data science",
                    path="python.txt",
                    type="text",
                    metadata={"language": "python"},
                ),
                Document(
                    id="doc2",
                    content="JavaScript is used for web development",
                    path="javascript.txt",
                    type="text",
                    metadata={"language": "javascript"},
                ),
            ]
        )

        db.populate(docs)

        result = db.query("programming language", n_results=1)

        assert isinstance(result, DocumentCollection)
        assert len(result.documents) > 0


def test_query_respects_n_results(llm):
    """Test that query() respects the n_results parameter."""
    with VectorDB(llm, name="test-query-limit") as db:
        docs = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Python programming",
                    path="p1.txt",
                    type="text",
                    metadata={},
                ),
                Document(
                    id="doc2",
                    content="Python tutorial",
                    path="p2.txt",
                    type="text",
                    metadata={},
                ),
                Document(
                    id="doc3",
                    content="Python guide",
                    path="p3.txt",
                    type="text",
                    metadata={},
                ),
            ]
        )

        db.populate(docs)

        result = db.query("Python", n_results=2)
        assert len(result.documents) <= 2


def test_query_returns_documents_with_metadata(llm):
    """Test that query() returns documents with correct metadata."""
    with VectorDB(llm, name="test-query-metadata") as db:
        docs = DocumentCollection(
            documents=[
                Document(
                    id="doc1",
                    content="Test content here",
                    path="test.txt",
                    type="text",
                    metadata={"section": "test"},
                )
            ]
        )

        db.populate(docs)

        result = db.query("test", n_results=1)

        assert len(result.documents) > 0
        doc = result.documents[0]
        assert doc.id == "doc1"
        assert doc.path == "test.txt"
        assert doc.type == "text"
        assert doc.metadata["section"] == "test"


def test_delete_removes_collection(llm):
    """Test that delete() removes the collection from ChromaDB."""
    db = VectorDB(llm, name="test-delete")

    docs = DocumentCollection(
        documents=[
            Document(
                id="doc1",
                content="Test content",
                path="test.txt",
                type="text",
                metadata={},
            )
        ]
    )

    db.populate(docs)

    collection_data = db.collection.get()
    assert len(collection_data["ids"]) == 1

    # Delete the collection
    db.delete()

    # Create a new db with the same name to verify collection is gone
    with VectorDB(llm, name="test-delete") as db_new:
        collection_data = db_new.collection.get()
        assert len(collection_data["ids"]) == 0


def test_delete_returns_none(llm):
    """Test that delete() returns None."""
    with VectorDB(llm, name="test-delete-return") as db:
        result = db.delete()
        assert result is None


def test_context_manager_calls_delete(llm):
    """Test that exiting with statement calls delete() automatically."""
    db = VectorDB(llm, name="test-context-manager")

    docs = DocumentCollection(
        documents=[
            Document(
                id="doc1",
                content="Test content",
                path="test.txt",
                type="text",
                metadata={},
            )
        ]
    )

    with db as context_db:
        assert context_db is db
        context_db.populate(docs)
        collection_data = context_db.collection.get()
        assert len(collection_data["ids"]) == 1

    # After exiting context, collection should be deleted
    with VectorDB(llm, name="test-context-manager") as db_new:
        collection_data = db_new.collection.get()
        assert len(collection_data["ids"]) == 0

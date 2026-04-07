from models.document import Document, DocumentCollection


def test_document_creation():
    """Test that Document can be created with all fields."""
    doc = Document(
        id="doc1",
        content="Test content",
        path="test.txt",
        type="text",
        metadata={"author": "Alice"},
    )

    assert doc.id == "doc1"
    assert doc.content == "Test content"
    assert doc.path == "test.txt"
    assert doc.type == "text"
    assert doc.metadata["author"] == "Alice"


def test_document_default_metrics():
    """Test that Document has empty metrics by default."""
    doc = Document(id="doc1", content="Test", path="test.txt", type="text", metadata={})
    assert doc.metrics == []


def test_document_add_with_another_document():
    """Test that Document + Document returns DocumentCollection."""
    doc1 = Document(
        id="doc1", content="Content 1", path="doc1.txt", type="text", metadata={}
    )
    doc2 = Document(
        id="doc2", content="Content 2", path="doc2.txt", type="text", metadata={}
    )

    result = doc1 + doc2

    assert isinstance(result, DocumentCollection)
    assert len(result.documents) == 2
    assert result.documents[0].id == "doc1"
    assert result.documents[1].id == "doc2"


def test_document_add_with_collection():
    """Test that Document + DocumentCollection returns DocumentCollection."""
    doc = Document(
        id="doc1", content="Content 1", path="doc1.txt", type="text", metadata={}
    )
    collection = DocumentCollection(
        documents=[
            Document(
                id="doc2",
                content="Content 2",
                path="doc2.txt",
                type="text",
                metadata={},
            ),
            Document(
                id="doc3",
                content="Content 3",
                path="doc3.txt",
                type="text",
                metadata={},
            ),
        ]
    )

    result = doc + collection

    assert isinstance(result, DocumentCollection)
    assert len(result.documents) == 3
    assert result.documents[0].id == "doc1"


def test_document_add_with_invalid_type():
    """Test that Document + invalid type raises exception."""
    doc = Document(
        id="doc1", content="Content", path="doc.txt", type="text", metadata={}
    )

    try:
        _ = doc + "invalid"
        assert False, "Should have raised exception"
    except Exception as e:
        assert "cannot add" in str(e)


def test_document_collection_creation():
    """Test that DocumentCollection can be created with documents."""
    docs = [
        Document(
            id="doc1", content="Content 1", path="doc1.txt", type="text", metadata={}
        ),
        Document(
            id="doc2", content="Content 2", path="doc2.txt", type="text", metadata={}
        ),
    ]

    collection = DocumentCollection(documents=docs)

    assert len(collection.documents) == 2
    assert collection.documents[0].id == "doc1"


def test_document_collection_default_empty():
    """Test that DocumentCollection defaults to empty list."""
    collection = DocumentCollection()
    assert collection.documents == []


def test_document_collection_from_docs():
    """Test DocumentCollection.from_docs() class method."""
    docs = [
        Document(
            id="doc1", content="Content 1", path="doc1.txt", type="text", metadata={}
        ),
        Document(
            id="doc2", content="Content 2", path="doc2.txt", type="text", metadata={}
        ),
    ]

    collection = DocumentCollection.from_docs(docs)

    assert len(collection.documents) == 2
    assert collection.documents[0].id == "doc1"


def test_document_collection_add_with_another_collection():
    """Test that DocumentCollection + DocumentCollection combines documents."""
    col1 = DocumentCollection(
        documents=[
            Document(
                id="doc1",
                content="Content 1",
                path="doc1.txt",
                type="text",
                metadata={},
            )
        ]
    )
    col2 = DocumentCollection(
        documents=[
            Document(
                id="doc2",
                content="Content 2",
                path="doc2.txt",
                type="text",
                metadata={},
            )
        ]
    )

    result = col1 + col2

    assert isinstance(result, DocumentCollection)
    assert len(result.documents) == 2


def test_document_collection_add_with_document():
    """Test that DocumentCollection + Document appends the document."""
    collection = DocumentCollection(
        documents=[
            Document(
                id="doc1",
                content="Content 1",
                path="doc1.txt",
                type="text",
                metadata={},
            )
        ]
    )
    doc = Document(
        id="doc2", content="Content 2", path="doc2.txt", type="text", metadata={}
    )

    result = collection + doc

    assert isinstance(result, DocumentCollection)
    assert len(result.documents) == 2
    assert result.documents[-1].id == "doc2"


def test_document_collection_add_with_invalid_type():
    """Test that DocumentCollection + invalid type raises exception."""
    collection = DocumentCollection()

    try:
        _ = collection + "invalid"
        assert False, "Should have raised exception"
    except Exception as e:
        assert "cannot add" in str(e)


def test_document_add_creates_independent_copy():
    """Test that adding documents doesn't modify originals."""
    doc1 = Document(
        id="doc1",
        content="Content 1",
        path="doc1.txt",
        type="text",
        metadata={"key": "value"},
    )
    doc2 = Document(
        id="doc2", content="Content 2", path="doc2.txt", type="text", metadata={}
    )

    collection = doc1 + doc2

    # Modify the result
    collection.documents[0].content = "Modified"

    # Original should be unchanged
    assert doc1.content == "Content 1"


def test_document_collection_add_creates_independent_copy():
    """Test that adding collections doesn't modify originals."""
    col1 = DocumentCollection(
        documents=[
            Document(
                id="doc1",
                content="Content 1",
                path="doc1.txt",
                type="text",
                metadata={},
            )
        ]
    )
    col2 = DocumentCollection(
        documents=[
            Document(
                id="doc2",
                content="Content 2",
                path="doc2.txt",
                type="text",
                metadata={},
            )
        ]
    )

    result = col1 + col2

    # Modify the result
    result.documents[0].content = "Modified"

    # Original should be unchanged
    assert col1.documents[0].content == "Content 1"

from models.query_context import RagContext, RagContextCollection, QueryContext


def test_rag_context_creation():
    """Test that RagContext can be created with all fields."""
    context = RagContext(
        id="ctx1", content="Test context content", metadata={"source": "document.pdf"}
    )

    assert context.id == "ctx1"
    assert context.content == "Test context content"
    assert context.metadata["source"] == "document.pdf"


def test_rag_context_default_metadata():
    """Test that RagContext defaults to empty metadata."""
    context = RagContext(id="ctx1", content="Content")
    assert context.metadata == {}


def test_rag_context_add_with_another_context():
    """Test that RagContext + RagContext returns RagContextCollection."""
    ctx1 = RagContext(id="ctx1", content="Content 1")
    ctx2 = RagContext(id="ctx2", content="Content 2")

    result = ctx1 + ctx2

    assert isinstance(result, RagContextCollection)
    assert len(result.contexts) == 2
    assert result.contexts[0].id == "ctx1"


def test_rag_context_add_with_collection():
    """Test that RagContext + RagContextCollection returns RagContextCollection."""
    ctx = RagContext(id="ctx1", content="Content 1")
    collection = RagContextCollection(
        contexts=[
            RagContext(id="ctx2", content="Content 2"),
            RagContext(id="ctx3", content="Content 3"),
        ]
    )

    result = ctx + collection

    assert isinstance(result, RagContextCollection)
    assert len(result.contexts) == 3
    assert result.contexts[0].id == "ctx1"


def test_rag_context_add_with_invalid_type():
    """Test that RagContext + invalid type raises exception."""
    ctx = RagContext(id="ctx1", content="Content")

    try:
        _ = ctx + "invalid"
        assert False, "Should have raised exception"
    except Exception as e:
        assert "cannot add" in str(e)


def test_rag_context_collection_creation():
    """Test that RagContextCollection can be created with contexts."""
    contexts = [
        RagContext(id="ctx1", content="Content 1"),
        RagContext(id="ctx2", content="Content 2"),
    ]

    collection = RagContextCollection(contexts=contexts)

    assert len(collection.contexts) == 2
    assert collection.contexts[0].id == "ctx1"


def test_rag_context_collection_default_empty():
    """Test that RagContextCollection defaults to empty list."""
    collection = RagContextCollection()
    assert collection.contexts == []


def test_rag_context_collection_from_contexts():
    """Test RagContextCollection.from_contexts() class method."""
    contexts = [
        RagContext(id="ctx1", content="Content 1"),
        RagContext(id="ctx2", content="Content 2"),
    ]

    collection = RagContextCollection.from_contexts(contexts)

    assert len(collection.contexts) == 2
    assert collection.contexts[0].id == "ctx1"


def test_rag_context_collection_add_with_another_collection():
    """Test that RagContextCollection + RagContextCollection combines contexts."""
    col1 = RagContextCollection(contexts=[RagContext(id="ctx1", content="Content 1")])
    col2 = RagContextCollection(contexts=[RagContext(id="ctx2", content="Content 2")])

    result = col1 + col2

    assert isinstance(result, RagContextCollection)
    assert len(result.contexts) == 2


def test_rag_context_collection_add_with_context():
    """Test that RagContextCollection + RagContext appends the context."""
    collection = RagContextCollection(
        contexts=[RagContext(id="ctx1", content="Content 1")]
    )
    ctx = RagContext(id="ctx2", content="Content 2")

    result = collection + ctx

    assert isinstance(result, RagContextCollection)
    assert len(result.contexts) == 2
    assert result.contexts[-1].id == "ctx2"


def test_rag_context_collection_add_with_invalid_type():
    """Test that RagContextCollection + invalid type raises exception."""
    collection = RagContextCollection()

    try:
        _ = collection + "invalid"
        assert False, "Should have raised exception"
    except Exception as e:
        assert "cannot add" in str(e)


def test_query_context_creation():
    """Test that QueryContext can be created with a question."""
    qc = QueryContext(question="What is Python?")

    assert qc.question == "What is Python?"
    assert len(qc.question_history) == 1


def test_query_context_default_fields():
    """Test that QueryContext initializes with default empty fields."""
    qc = QueryContext(question="Test question")

    assert isinstance(qc.context, RagContextCollection)
    assert qc.context.contexts == []


def test_query_context_question_getter():
    """Test that question property returns the latest question."""
    qc = QueryContext(question="Question 1")
    qc.question = "Question 2"

    assert qc.question == "Question 2"


def test_query_context_with_context():
    """Test that QueryContext can be initialized with context data."""
    context = RagContextCollection(
        contexts=[RagContext(id="ctx1", content="Some context")]
    )

    qc = QueryContext(question="Question", context=context)

    assert qc.question == "Question"
    assert len(qc.context.contexts) == 1


def test_query_context_question_history_independence():
    """Test that multiple QueryContext instances don't share question_history."""
    qc1 = QueryContext(question="Q1")
    qc2 = QueryContext(question="Q2")

    qc1.question = "Q1-Updated"

    assert qc2.question == "Q2"

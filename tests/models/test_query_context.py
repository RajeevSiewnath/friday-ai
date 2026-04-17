import pytest
from friday.models.query_context import RagContext, RagContextCollection, QueryContext


class TestRagContext:
    def test_rag_context_initialization(self):
        context = RagContext(id="doc1", content="test content")
        assert context.id == "doc1"
        assert context.content == "test content"
        assert context.metadata == {}

    def test_rag_context_with_metadata(self):
        metadata = {"source": "file.txt", "page": 1}
        context = RagContext(id="doc1", content="content", metadata=metadata)
        assert context.metadata == {"source": "file.txt", "page": 1}

    def test_rag_context_add_to_rag_context(self):
        context1 = RagContext(id="doc1", content="content1")
        context2 = RagContext(id="doc2", content="content2")

        result = context1 + context2
        assert isinstance(result, RagContextCollection)
        assert len(result.contexts) == 2
        assert result.contexts[0].id == "doc1"
        assert result.contexts[1].id == "doc2"

    def test_rag_context_add_to_collection(self):
        context1 = RagContext(id="doc1", content="content1")
        collection = RagContextCollection(
            contexts=[RagContext(id="doc2", content="content2")]
        )

        result = context1 + collection
        assert isinstance(result, RagContextCollection)
        assert len(result.contexts) == 2
        assert result.contexts[0].id == "doc1"
        assert result.contexts[1].id == "doc2"

    def test_rag_context_add_invalid_type_raises_exception(self):
        context = RagContext(id="doc1", content="content")
        with pytest.raises(Exception):
            _ = context + "invalid"

    def test_rag_context_original_unmodified_after_add(self):
        context1 = RagContext(id="doc1", content="content1", metadata={"key": "value"})
        context2 = RagContext(id="doc2", content="content2")

        result = context1 + context2
        # Ensure original is unchanged
        assert context1.id == "doc1"
        assert len(result.contexts) == 2


class TestRagContextCollection:
    def test_collection_initialization(self):
        collection = RagContextCollection()
        assert collection.contexts == []

    def test_collection_initialization_with_contexts(self):
        contexts = [
            RagContext(id="doc1", content="content1"),
            RagContext(id="doc2", content="content2"),
        ]
        collection = RagContextCollection(contexts=contexts)
        assert len(collection.contexts) == 2

    def test_collection_from_contexts(self):
        contexts = [
            RagContext(id="doc1", content="content1"),
            RagContext(id="doc2", content="content2"),
        ]
        collection = RagContextCollection.from_contexts(contexts)
        assert len(collection.contexts) == 2
        assert collection.contexts[0].id == "doc1"

    def test_collection_add_to_collection(self):
        collection1 = RagContextCollection(
            contexts=[RagContext(id="doc1", content="content1")]
        )
        collection2 = RagContextCollection(
            contexts=[RagContext(id="doc2", content="content2")]
        )

        result = collection1 + collection2
        assert isinstance(result, RagContextCollection)
        assert len(result.contexts) == 2

    def test_collection_add_to_context(self):
        collection = RagContextCollection(
            contexts=[RagContext(id="doc1", content="content1")]
        )
        context = RagContext(id="doc2", content="content2")

        result = collection + context
        assert isinstance(result, RagContextCollection)
        assert len(result.contexts) == 2
        assert result.contexts[1].id == "doc2"

    def test_collection_add_invalid_type_raises_exception(self):
        collection = RagContextCollection()
        with pytest.raises(Exception):
            _ = collection + "invalid"

    def test_collection_original_unmodified_after_add(self):
        collection1 = RagContextCollection(
            contexts=[RagContext(id="doc1", content="content1")]
        )
        collection2 = RagContextCollection(
            contexts=[RagContext(id="doc2", content="content2")]
        )

        result = collection1 + collection2
        assert len(collection1.contexts) == 1
        assert len(result.contexts) == 2


class TestQueryContext:
    def test_query_context_initialization_empty(self):
        context = QueryContext()
        assert context.question is None
        assert context.question_history == []
        assert isinstance(context.context, RagContextCollection)

    def test_query_context_initialization_with_question(self):
        context = QueryContext(question="What is AI?")
        assert context.question == "What is AI?"
        assert context.question_history == ["What is AI?"]

    def test_query_context_question_property_getter(self):
        context = QueryContext()
        context.question_history = ["first", "second", "third"]
        assert context.question == "third"

    def test_query_context_question_property_setter(self):
        context = QueryContext()
        context.question = "first question"
        context.question = "second question"

        assert context.question == "second question"
        assert len(context.question_history) == 2
        assert context.question_history[0] == "first question"
        assert context.question_history[1] == "second question"

    def test_query_context_question_history_initialization(self):
        context = QueryContext(question="initial")
        assert context.question_history == ["initial"]

        context.question = "updated"
        assert context.question_history == ["initial", "updated"]

    def test_query_context_with_context_data(self):
        rag_context = RagContextCollection(
            contexts=[RagContext(id="doc1", content="content")]
        )
        context = QueryContext(question="test", context=rag_context)

        assert context.question == "test"
        assert len(context.context.contexts) == 1

    def test_query_context_question_history_empty_returns_none(self):
        context = QueryContext()
        assert context.question is None

    def test_query_context_multiple_question_updates(self):
        context = QueryContext(question="q1")
        context.question = "q2"
        context.question = "q3"
        context.question = "q4"

        assert context.question == "q4"
        assert len(context.question_history) == 4

    def test_query_context_preserves_context_on_question_update(self):
        rag_context = RagContextCollection(
            contexts=[RagContext(id="doc1", content="content1")]
        )
        context = QueryContext(question="q1", context=rag_context)
        context.question = "q2"

        assert context.question == "q2"
        assert len(context.context.contexts) == 1

    def test_query_context_serialization(self):
        context = QueryContext(question="test")
        data = context.model_dump()

        assert data["question_history"] == ["test"]
        assert "context" in data

    def test_query_context_deserialization(self):
        data = {
            "question_history": ["q1", "q2"],
            "context": {"contexts": []},
        }
        context = QueryContext(**data)

        assert context.question == "q2"
        assert len(context.question_history) == 2

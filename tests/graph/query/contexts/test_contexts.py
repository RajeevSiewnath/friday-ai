import pytest
from unittest.mock import MagicMock
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.contexts.user_context import UserContext
from friday.graph.query.contexts.vector_db_context import VectorDBContext
from friday.graph.query.contexts.rag_context import RagContext
from friday.core.vector_db import VectorQueryOutput


class TestLLMContext:
    def test_llm_context_initialization(self):
        """Test LLMContext initialization."""
        mock_llm = MagicMock()
        context = LLMContext(llm=mock_llm)

        assert context.llm == mock_llm

    def test_llm_context_with_real_llm_type(self):
        """Test LLMContext attributes."""
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        context = LLMContext(llm=mock_llm)

        assert hasattr(context, "llm")
        assert context.llm.model == "gpt-4"

    def test_llm_context_is_dataclass(self):
        """Test that LLMContext is a dataclass."""
        assert hasattr(LLMContext, "__dataclass_fields__")
        assert "llm" in LLMContext.__dataclass_fields__

    def test_llm_context_equality(self):
        """Test LLMContext equality comparison."""
        mock_llm1 = MagicMock()
        mock_llm1.id = "llm1"
        mock_llm2 = MagicMock()
        mock_llm2.id = "llm1"

        # Same LLM object should be equal
        context1 = LLMContext(llm=mock_llm1)
        context2 = LLMContext(llm=mock_llm1)
        assert context1 == context2

    def test_llm_context_repr(self):
        """Test LLMContext string representation."""
        mock_llm = MagicMock()
        context = LLMContext(llm=mock_llm)

        repr_str = repr(context)
        assert "LLMContext" in repr_str

    def test_llm_context_field_access(self):
        """Test accessing LLMContext fields."""
        mock_llm = MagicMock()
        context = LLMContext(llm=mock_llm)

        # Direct attribute access
        assert context.llm is mock_llm

        # Dict-like access through dataclass
        assert hasattr(context, "__dataclass_fields__")


class TestUserContext:
    def test_user_context_initialization(self):
        """Test UserContext initialization."""
        context = UserContext(user="john_doe", user_context="Administrator")

        assert context.user == "john_doe"
        assert context.user_context == "Administrator"

    def test_user_context_with_empty_strings(self):
        """Test UserContext with empty strings."""
        context = UserContext(user="", user_context="")

        assert context.user == ""
        assert context.user_context == ""

    def test_user_context_with_special_characters(self):
        """Test UserContext with special characters in user."""
        context = UserContext(
            user="user@example.com", user_context="Admin (Full Access)"
        )

        assert context.user == "user@example.com"
        assert context.user_context == "Admin (Full Access)"

    def test_user_context_is_dataclass(self):
        """Test that UserContext is a dataclass."""
        assert hasattr(UserContext, "__dataclass_fields__")
        assert "user" in UserContext.__dataclass_fields__
        assert "user_context" in UserContext.__dataclass_fields__

    def test_user_context_equality(self):
        """Test UserContext equality comparison."""
        context1 = UserContext(user="alice", user_context="Editor")
        context2 = UserContext(user="alice", user_context="Editor")
        context3 = UserContext(user="bob", user_context="Editor")

        assert context1 == context2
        assert context1 != context3

    def test_user_context_repr(self):
        """Test UserContext string representation."""
        context = UserContext(user="bob", user_context="Viewer")

        repr_str = repr(context)
        assert "UserContext" in repr_str
        assert "bob" in repr_str

    def test_user_context_with_unicode(self):
        """Test UserContext with unicode characters."""
        context = UserContext(
            user="用户", user_context="管理员"
        )

        assert context.user == "用户"
        assert context.user_context == "管理员"

    def test_user_context_field_access(self):
        """Test accessing UserContext fields."""
        context = UserContext(user="user1", user_context="context1")

        assert context.user == "user1"
        assert context.user_context == "context1"


class TestVectorDBContext:
    def test_vector_db_context_initialization(self):
        """Test VectorDBContext initialization."""
        mock_vector_db = MagicMock()
        context = VectorDBContext(vector_db=mock_vector_db)

        assert context.vector_db == mock_vector_db

    def test_vector_db_context_with_mock_db(self):
        """Test VectorDBContext with mock database operations."""
        mock_vector_db = MagicMock()
        mock_vector_db.chroma = MagicMock()
        context = VectorDBContext(vector_db=mock_vector_db)

        assert hasattr(context, "vector_db")
        assert context.vector_db.chroma is not None

    def test_vector_db_context_is_dataclass(self):
        """Test that VectorDBContext is a dataclass."""
        assert hasattr(VectorDBContext, "__dataclass_fields__")
        assert "vector_db" in VectorDBContext.__dataclass_fields__

    def test_vector_db_context_equality(self):
        """Test VectorDBContext equality comparison."""
        mock_db1 = MagicMock()
        mock_db1.id = "db1"

        context1 = VectorDBContext(vector_db=mock_db1)
        context2 = VectorDBContext(vector_db=mock_db1)

        assert context1 == context2

    def test_vector_db_context_repr(self):
        """Test VectorDBContext string representation."""
        mock_db = MagicMock()
        context = VectorDBContext(vector_db=mock_db)

        repr_str = repr(context)
        assert "VectorDBContext" in repr_str

    def test_vector_db_context_field_access(self):
        """Test accessing VectorDBContext fields."""
        mock_db = MagicMock()
        context = VectorDBContext(vector_db=mock_db)

        assert context.vector_db is mock_db


class TestRagContext:
    def test_rag_context_initialization_empty(self):
        """Test RagContext initialization with empty data."""
        context = RagContext(rag_data={})

        assert context.rag_data == {}

    def test_rag_context_with_single_set(self):
        """Test RagContext with single result set."""
        outputs = [
            VectorQueryOutput(id="doc1", metadata={}, document="content1"),
            VectorQueryOutput(id="doc2", metadata={}, document="content2"),
        ]
        context = RagContext(rag_data={"set1": outputs})

        assert "set1" in context.rag_data
        assert len(context.rag_data["set1"]) == 2

    def test_rag_context_with_multiple_sets(self):
        """Test RagContext with multiple result sets."""
        outputs1 = [VectorQueryOutput(id="doc1", metadata={}, document="content1")]
        outputs2 = [VectorQueryOutput(id="doc2", metadata={}, document="content2")]
        outputs3 = [VectorQueryOutput(id="doc3", metadata={}, document="content3")]

        rag_data = {"retrieval": outputs1, "reranked": outputs2, "final": outputs3}
        context = RagContext(rag_data=rag_data)

        assert len(context.rag_data) == 3
        assert len(context.rag_data["retrieval"]) == 1
        assert len(context.rag_data["reranked"]) == 1
        assert len(context.rag_data["final"]) == 1

    def test_rag_context_is_dataclass(self):
        """Test that RagContext is a dataclass."""
        assert hasattr(RagContext, "__dataclass_fields__")
        assert "rag_data" in RagContext.__dataclass_fields__

    def test_rag_context_equality(self):
        """Test RagContext equality comparison."""
        outputs = [VectorQueryOutput(id="doc1", metadata={}, document="content")]
        rag_data = {"set1": outputs}

        context1 = RagContext(rag_data=rag_data)
        context2 = RagContext(rag_data=rag_data)

        assert context1 == context2

    def test_rag_context_repr(self):
        """Test RagContext string representation."""
        context = RagContext(rag_data={})

        repr_str = repr(context)
        assert "RagContext" in repr_str

    def test_rag_context_with_metadata(self):
        """Test RagContext preserves metadata in outputs."""
        metadata = {"source": "file.txt", "page": 5}
        outputs = [VectorQueryOutput(id="doc1", metadata=metadata, document="content")]
        context = RagContext(rag_data={"set1": outputs})

        assert context.rag_data["set1"][0].metadata == metadata

    def test_rag_context_with_complex_document_content(self):
        """Test RagContext with complex document content."""
        long_content = "This is a very long document " * 100
        outputs = [
            VectorQueryOutput(id="doc1", metadata={}, document=long_content)
        ]
        context = RagContext(rag_data={"set1": outputs})

        assert len(context.rag_data["set1"][0].document) == len(long_content)

    def test_rag_context_field_access(self):
        """Test accessing RagContext fields."""
        rag_data = {
            "results": [
                VectorQueryOutput(id="doc1", metadata={}, document="content")
            ]
        }
        context = RagContext(rag_data=rag_data)

        assert context.rag_data == rag_data

    def test_rag_context_multiple_documents_per_set(self):
        """Test RagContext with multiple documents per set."""
        outputs = [
            VectorQueryOutput(id=f"doc{i}", metadata={"index": i}, document=f"content{i}")
            for i in range(10)
        ]
        context = RagContext(rag_data={"large_set": outputs})

        assert len(context.rag_data["large_set"]) == 10
        for i, output in enumerate(context.rag_data["large_set"]):
            assert output.id == f"doc{i}"
            assert output.metadata["index"] == i

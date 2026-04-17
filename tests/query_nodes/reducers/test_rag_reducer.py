import pytest
from friday.query_nodes.reducers.rag_reducer import (
    rag_reducer,
    RagReducerReplaceAction,
)
from friday.core.vector_db import VectorQueryOutput


class TestRagReducer:
    def test_reducer_adds_new_key(self):
        """Test reducer adds new keys."""
        left = {"set1": []}
        right = {
            "set2": [VectorQueryOutput(id="doc1", metadata={}, document="content")]
        }

        result = rag_reducer(left, right)

        assert "set1" in result
        assert "set2" in result

    def test_reducer_extends_existing_key(self):
        """Test reducer extends existing keys."""
        left = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="content1")]
        }
        right = {
            "set1": [VectorQueryOutput(id="doc2", metadata={}, document="content2")]
        }

        result = rag_reducer(left, right)

        assert len(result["set1"]) == 2

    def test_reducer_empty_left(self):
        """Test reducer with empty left."""
        left = {}
        right = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="content")]
        }

        result = rag_reducer(left, right)

        assert "set1" in result
        assert len(result["set1"]) == 1

    def test_reducer_empty_right(self):
        """Test reducer with empty right."""
        left = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="content")]
        }
        right = {}

        result = rag_reducer(left, right)

        assert "set1" in result
        assert len(result["set1"]) == 1

    def test_reducer_both_empty(self):
        """Test reducer with both empty."""
        left = {}
        right = {}

        result = rag_reducer(left, right)

        assert result == {}

    def test_reducer_multiple_keys(self):
        """Test reducer with multiple keys."""
        left = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="content1")],
            "set2": [VectorQueryOutput(id="doc2", metadata={}, document="content2")],
        }
        right = {
            "set1": [VectorQueryOutput(id="doc3", metadata={}, document="content3")],
            "set3": [VectorQueryOutput(id="doc4", metadata={}, document="content4")],
        }

        result = rag_reducer(left, right)

        assert len(result) == 3
        assert len(result["set1"]) == 2
        assert len(result["set2"]) == 1
        assert len(result["set3"]) == 1

    def test_reducer_preserves_metadata(self):
        """Test reducer preserves metadata."""
        metadata = {"source": "file.txt", "page": 1}
        output = VectorQueryOutput(id="doc1", metadata=metadata, document="content")

        left = {}
        right = {"set1": [output]}

        result = rag_reducer(left, right)

        assert result["set1"][0].metadata == metadata

    def test_reducer_deepcopy_isolation(self):
        """Test reducer doesn't modify original left."""
        output = VectorQueryOutput(id="doc1", metadata={}, document="content1")
        left = {"set1": [output]}
        original_len = len(left)

        right = {
            "set2": [VectorQueryOutput(id="doc2", metadata={}, document="content2")]
        }

        result = rag_reducer(left, right)

        # Original left should be unchanged
        assert len(left) == original_len
        assert "set2" not in left

    def test_reducer_with_complex_metadata(self):
        """Test reducer with complex metadata."""
        metadata = {
            "source": "document.pdf",
            "page": 5,
            "chapter": "Introduction",
            "nested": {"key": "value"},
        }
        output = VectorQueryOutput(
            id="doc1",
            metadata=metadata,
            document="complex content",
        )

        left = {}
        right = {"set1": [output]}

        result = rag_reducer(left, right)

        assert result["set1"][0].metadata == metadata
        assert result["set1"][0].metadata["nested"]["key"] == "value"

    def test_reducer_preserves_document_content(self):
        """Test reducer preserves full document content."""
        long_content = "This is content " * 100
        output = VectorQueryOutput(id="doc1", metadata={}, document=long_content)

        left = {}
        right = {"set1": [output]}

        result = rag_reducer(left, right)

        assert result["set1"][0].document == long_content

    def test_reducer_multiple_documents_per_set(self):
        """Test reducer with multiple documents per set."""
        outputs = [
            VectorQueryOutput(
                id=f"doc{i}",
                metadata={"index": i},
                document=f"content{i}"
            )
            for i in range(5)
        ]

        left = {}
        right = {"set1": outputs}

        result = rag_reducer(left, right)

        assert len(result["set1"]) == 5

    def test_reducer_multiple_sequential_extends(self):
        """Test multiple sequential extend operations."""
        left = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="content1")]
        }

        right1 = {
            "set1": [VectorQueryOutput(id="doc2", metadata={}, document="content2")]
        }
        result = rag_reducer(left, right1)
        assert len(result["set1"]) == 2

        right2 = {
            "set1": [VectorQueryOutput(id="doc3", metadata={}, document="content3")]
        }
        result = rag_reducer(result, right2)
        assert len(result["set1"]) == 3

    def test_reducer_create_new_key_if_not_exists(self):
        """Test reducer creates new key if not in output."""
        left = {}
        right = {
            "new_key": [VectorQueryOutput(id="doc1", metadata={}, document="content")]
        }

        result = rag_reducer(left, right)

        assert "new_key" in result

    def test_reducer_extend_empty_list(self):
        """Test reducer extends empty list."""
        left = {"set1": []}
        right = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="content")]
        }

        result = rag_reducer(left, right)

        assert len(result["set1"]) == 1

    def test_reducer_with_unicode_content(self):
        """Test reducer with unicode content."""
        output = VectorQueryOutput(id="doc1", metadata={}, document="你好世界")

        left = {}
        right = {"set1": [output]}

        result = rag_reducer(left, right)

        assert result["set1"][0].document == "你好世界"

    def test_reducer_with_special_characters(self):
        """Test reducer with special characters."""
        output = VectorQueryOutput(
            id="doc@#$",
            metadata={"key": "value!@#$"},
            document="content%^&"
        )

        left = {}
        right = {"set1": [output]}

        result = rag_reducer(left, right)

        assert result["set1"][0].id == "doc@#$"
        assert result["set1"][0].metadata["key"] == "value!@#$"

    def test_replace_action_initialization(self):
        """Test RagReducerReplaceAction initialization."""
        action = RagReducerReplaceAction()
        assert isinstance(action, RagReducerReplaceAction)

    def test_reducer_preserves_vector_output_structure(self):
        """Test reducer preserves VectorQueryOutput structure."""
        output = VectorQueryOutput(
            id="unique_id",
            metadata={"type": "document"},
            document="full content"
        )

        left = {}
        right = {"results": [output]}

        result = rag_reducer(left, right)

        retrieved = result["results"][0]
        assert isinstance(retrieved, VectorQueryOutput)
        assert retrieved.id == "unique_id"
        assert retrieved.metadata["type"] == "document"
        assert retrieved.document == "full content"

    def test_reducer_multiple_keys_same_right(self):
        """Test reducer with multiple keys in single right dict."""
        left = {}
        right = {
            "retrieval": [VectorQueryOutput(id="doc1", metadata={}, document="content1")],
            "reranked": [VectorQueryOutput(id="doc2", metadata={}, document="content2")],
            "final": [VectorQueryOutput(id="doc3", metadata={}, document="content3")],
        }

        result = rag_reducer(left, right)

        assert len(result) == 3
        assert len(result["retrieval"]) == 1
        assert len(result["reranked"]) == 1
        assert len(result["final"]) == 1

    def test_reducer_extend_with_duplicate_ids(self):
        """Test reducer extends even with duplicate ids."""
        left = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="original")]
        }
        right = {
            "set1": [VectorQueryOutput(id="doc1", metadata={}, document="new")]
        }

        result = rag_reducer(left, right)

        # Should have both (extends, doesn't replace)
        assert len(result["set1"]) == 2

    def test_reducer_large_document_set(self):
        """Test reducer with large document set."""
        outputs = [
            VectorQueryOutput(id=f"doc{i}", metadata={}, document=f"content{i}")
            for i in range(1000)
        ]

        left = {}
        right = {"large_set": outputs}

        result = rag_reducer(left, right)

        assert len(result["large_set"]) == 1000

    def test_reducer_preserves_list_order(self):
        """Test reducer preserves list order."""
        left = {
            "set1": [
                VectorQueryOutput(id="doc1", metadata={}, document="first"),
                VectorQueryOutput(id="doc2", metadata={}, document="second"),
            ]
        }
        right = {
            "set1": [
                VectorQueryOutput(id="doc3", metadata={}, document="third"),
            ]
        }

        result = rag_reducer(left, right)

        assert result["set1"][0].id == "doc1"
        assert result["set1"][1].id == "doc2"
        assert result["set1"][2].id == "doc3"

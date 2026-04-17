import pytest
from unittest.mock import MagicMock, patch
from chromadb.config import Settings
from friday.core.vector_db import VectorDB, VectorDBCollection, VectorQueryInput, VectorQueryOutput


class TestVectorQueryModels:
    def test_vector_query_output_creation(self):
        output = VectorQueryOutput(
            id="test_id", metadata={"key": "value"}, document="test document"
        )

        assert output.id == "test_id"
        assert output.metadata == {"key": "value"}
        assert output.document == "test document"

    def test_vector_query_input_creation(self):
        input_data = VectorQueryInput(
            id="test_id",
            metadata={"key": "value"},
            document="test document",
            embedding=[0.1, 0.2, 0.3],
        )

        assert input_data.id == "test_id"
        assert input_data.metadata == {"key": "value"}
        assert input_data.document == "test document"
        assert input_data.embedding == [0.1, 0.2, 0.3]


@pytest.fixture
def mock_chroma_client():
    return MagicMock()


@pytest.fixture
def mock_collection():
    collection = MagicMock()
    collection.name = "test_collection"
    collection.get.return_value = {"ids": []}
    return collection


class TestVectorDB:
    def test_initialization_default(self, mock_chroma_client):
        with patch("friday.core.vector_db.Client", return_value=mock_chroma_client):
            db = VectorDB()

            assert db.chroma == mock_chroma_client
            assert db._VectorDB__collections == []

    def test_initialization_custom_settings(self, mock_chroma_client):
        custom_settings = Settings(is_persistent=False)

        with patch("friday.core.vector_db.Client", return_value=mock_chroma_client):
            db = VectorDB(settings=custom_settings)

            assert db.chroma == mock_chroma_client

    def test_getitem_creates_new_collection(self, mock_chroma_client, mock_collection):
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("friday.core.vector_db.Client", return_value=mock_chroma_client):
            db = VectorDB()
            collection = db["test_collection"]

            assert isinstance(collection, VectorDBCollection)
            assert mock_chroma_client.get_or_create_collection.called
            assert len(db._VectorDB__collections) == 1

    def test_getitem_returns_existing_collection(self, mock_chroma_client, mock_collection):
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("friday.core.vector_db.Client", return_value=mock_chroma_client):
            db = VectorDB()
            collection1 = db["test_collection"]
            collection2 = db["test_collection"]

            assert collection1 == collection2
            assert mock_chroma_client.get_or_create_collection.call_count == 1

    def test_delete_collection(self, mock_chroma_client, mock_collection):
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("friday.core.vector_db.Client", return_value=mock_chroma_client):
            db = VectorDB()
            collection = db["test_collection"]

            assert len(db._VectorDB__collections) == 1

            db.delete(collection)

            assert len(db._VectorDB__collections) == 0
            assert mock_chroma_client.delete_collection.called


class TestVectorDBCollection:
    @pytest.fixture
    def vector_db(self, mock_chroma_client):
        with patch("friday.core.vector_db.Client", return_value=mock_chroma_client):
            return VectorDB()

    def test_initialization(self, vector_db, mock_collection):
        collection = VectorDBCollection(vector_db, mock_collection)

        assert collection.vector_db == vector_db
        assert collection.collection == mock_collection

    def test_populate_empty_collection(self, vector_db, mock_collection):
        mock_collection.get.return_value = {"ids": []}

        collection = VectorDBCollection(vector_db, mock_collection)

        input_data = [
            VectorQueryInput(
                id="id1",
                metadata={"type": "test"},
                document="doc1",
                embedding=[0.1, 0.2],
            ),
            VectorQueryInput(
                id="id2",
                metadata={"type": "test"},
                document="doc2",
                embedding=[0.3, 0.4],
            ),
        ]

        result = collection.populate(input_data)

        assert result == collection  # Returns self for chaining
        mock_collection.upsert.assert_called_once()

        call_args = mock_collection.upsert.call_args
        assert call_args[1]["ids"] == ["id1", "id2"]
        assert call_args[1]["documents"] == ["doc1", "doc2"]
        assert call_args[1]["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert call_args[1]["metadatas"] == [{"type": "test"}, {"type": "test"}]

    def test_populate_with_force_flag(self, vector_db, mock_collection):
        mock_collection.get.return_value = {"ids": ["existing"]}

        collection = VectorDBCollection(vector_db, mock_collection)

        input_data = [
            VectorQueryInput(
                id="id1", metadata={}, document="doc1", embedding=[0.1, 0.2]
            )
        ]

        collection.populate(input_data, force=True)

        mock_collection.upsert.assert_called_once()

    def test_populate_skips_non_empty_collection(self, vector_db, mock_collection):
        mock_collection.get.return_value = {"ids": ["existing_id"]}

        collection = VectorDBCollection(vector_db, mock_collection)

        input_data = [
            VectorQueryInput(
                id="id1", metadata={}, document="doc1", embedding=[0.1, 0.2]
            )
        ]

        collection.populate(input_data, force=False)

        mock_collection.upsert.assert_not_called()

    def test_populate_with_clear_flag(self, vector_db, mock_chroma_client, mock_collection):
        mock_collection.get.return_value = {"ids": []}
        mock_chroma_client.create_collection.return_value = mock_collection

        with patch("friday.core.vector_db.Client", return_value=mock_chroma_client):
            db = VectorDB()
            collection = VectorDBCollection(db, mock_collection)

            input_data = [
                VectorQueryInput(
                    id="id1", metadata={}, document="doc1", embedding=[0.1, 0.2]
                )
            ]

            collection.populate(input_data, clear=True)

            mock_chroma_client.delete_collection.assert_called()
            mock_chroma_client.create_collection.assert_called()

    def test_query(self, vector_db, mock_collection):
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "metadatas": [[{"type": "a"}, {"type": "b"}]],
            "documents": [["doc1", "doc2"]],
        }

        collection = VectorDBCollection(vector_db, mock_collection)
        results = collection.query(embedding=[0.1, 0.2], n_results=2)

        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].metadata == {"type": "a"}
        assert results[0].document == "doc1"
        assert results[1].id == "id2"
        assert results[1].metadata == {"type": "b"}
        assert results[1].document == "doc2"

        mock_collection.query.assert_called_once_with(
            query_embeddings=[0.1, 0.2], n_results=2
        )

    def test_query_default_n_results(self, vector_db, mock_collection):
        mock_collection.query.return_value = {
            "ids": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }

        collection = VectorDBCollection(vector_db, mock_collection)
        collection.query(embedding=[0.1, 0.2])

        call_args = mock_collection.query.call_args
        assert call_args[1]["n_results"] == 10

    def test_collection_delete_through_vector_db(self, vector_db, mock_collection):
        # Test the normal flow: get collection from VectorDB, then delete it
        vector_db.chroma.get_or_create_collection = MagicMock(return_value=mock_collection)

        collection = vector_db["test_collection"]
        assert len(vector_db._VectorDB__collections) == 1

        vector_db.delete(collection)
        assert len(vector_db._VectorDB__collections) == 0

    def test_populate_returns_self_for_chaining(self, vector_db, mock_collection):
        mock_collection.get.return_value = {"ids": []}

        collection = VectorDBCollection(vector_db, mock_collection)
        input_data = [
            VectorQueryInput(
                id="id1", metadata={}, document="doc1", embedding=[0.1, 0.2]
            )
        ]

        result = collection.populate(input_data)

        assert result == collection

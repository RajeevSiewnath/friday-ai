import pytest
import pytest_asyncio
from unittest.mock import MagicMock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    return MagicMock()


@pytest.fixture
def mock_chroma_client():
    """Mock Chroma client for testing."""
    return MagicMock()

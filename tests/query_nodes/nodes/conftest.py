import pytest
from unittest.mock import MagicMock


@pytest.fixture
def make_runtime():
    def _make(context: dict):
        runtime_mock = MagicMock()
        for key, value in context.items():
            setattr(runtime_mock.context, key, value)
        return runtime_mock

    return _make

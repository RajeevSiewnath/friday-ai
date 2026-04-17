import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from friday.core.llm import LLM, LLMOpenAISDKConstructorArgs
from friday.core.tool_shed import ToolShed
from openai.types.responses import Response
from openai.types.file_object import FileObject


@pytest.fixture
def mock_tool():
    def test_tool(x: int) -> int:
        """Test tool."""
        return x * 2

    return test_tool


class TestLLMInitialization:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("friday.core.llm.AsyncOpenAI")
    def test_initialization_default(self, mock_openai):
        llm = LLM()

        assert llm.model == "gpt-4.1-nano"
        assert llm.embedding_model == "text-embedding-3-small"
        assert isinstance(llm.tool_shed, ToolShed)
        mock_openai.assert_called_once()

    @patch("friday.core.llm.AsyncOpenAI")
    def test_initialization_custom_api_key(self, mock_openai):
        llm = LLM(api_key="custom_key")

        mock_openai.assert_called_once()
        assert mock_openai.call_args[1]["api_key"] == "custom_key"

    @patch("friday.core.llm.AsyncOpenAI")
    def test_initialization_custom_models(self, mock_openai):
        llm = LLM(api_key="test", model="gpt-4", embedding_model="text-embedding-3-large")

        assert llm.model == "gpt-4"
        assert llm.embedding_model == "text-embedding-3-large"

    @patch("friday.core.llm.AsyncOpenAI")
    def test_initialization_with_tool_shed(self, mock_openai, mock_tool):
        tool_shed = ToolShed(mock_tool)
        llm = LLM(api_key="test", tool_shed=tool_shed)

        assert llm.tool_shed == tool_shed

    @patch("friday.core.llm.AsyncOpenAI")
    def test_initialization_with_kwargs(self, mock_openai):
        llm = LLM(
            api_key="test",
            organization="org-123",
            project="proj-456",
            base_url="https://custom.openai.com",
        )

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["organization"] == "org-123"
        assert call_kwargs["project"] == "proj-456"
        assert call_kwargs["base_url"] == "https://custom.openai.com"


class TestLLMInvoke:
    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_invoke_with_string_response(self, mock_openai_class, mock_tool):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock(spec=Response)
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        tool_shed = ToolShed(mock_tool)
        llm = LLM(api_key="test", tool_shed=tool_shed)
        result = await llm.invoke([{"role": "user", "content": "test"}])

        assert result == mock_response
        mock_client.responses.create.assert_called_once()

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4.1-nano"
        assert call_kwargs["input"] == [{"role": "user", "content": "test"}]

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_invoke_with_no_tools(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock(spec=Response)
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        llm = LLM(api_key="test")
        await llm.invoke([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["tools"] == []

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_invoke_with_parsed_response(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_client.responses.parse = AsyncMock(return_value=mock_response)

        class CustomFormat(dict):
            pass

        llm = LLM(api_key="test")
        result = await llm.invoke([{"role": "user", "content": "test"}], response_format=CustomFormat)

        assert result == mock_response
        mock_client.responses.parse.assert_called_once()


class TestLLMStream:
    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_stream_basic(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        async def async_event_generator():
            yield MagicMock(type="response.output_item.added", item=MagicMock(model_dump=lambda: {"type": "text", "text": "", "content": ""}))
            yield MagicMock(type="response.output_text.delta", delta="Hello")
            yield MagicMock(type="response.completed", response=MagicMock(output=[MagicMock(status="completed")]))

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)
        mock_stream.__aiter__ = lambda self: async_event_generator()

        mock_client.responses.stream = MagicMock(return_value=mock_stream)

        llm = LLM(api_key="test")
        results = []
        async for chunk in llm.stream([{"role": "user", "content": "test"}]):
            results.append(chunk)

        assert len(results) > 0

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_stream_with_function_call(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        async def async_event_generator():
            event1 = MagicMock(type="response.output_item.added")
            event1.item = MagicMock()
            event1.item.model_dump = lambda: {"type": "function_call", "arguments": ""}
            yield event1

            event2 = MagicMock(type="response.function_call_arguments.delta")
            event2.delta = '{"x": 1'
            yield event2

            event3 = MagicMock(type="response.function_call_arguments.delta")
            event3.delta = '0}'
            yield event3

            yield MagicMock(type="response.function_call_arguments.done")

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)
        mock_stream.__aiter__ = lambda self: async_event_generator()

        mock_client.responses.stream = MagicMock(return_value=mock_stream)

        llm = LLM(api_key="test")
        results = []
        async for chunk in llm.stream([{"role": "user", "content": "test"}]):
            results.append(chunk)

        assert len(results) > 0


class TestLLMEmbeddings:
    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_embedding_single(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        llm = LLM(api_key="test")
        result = await llm.embedding("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_embeddings_multiple(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        llm = LLM(api_key="test")
        result = await llm.embeddings(["text1", "text2"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["text1", "text2"]
        )


class TestLLMFileOperations:
    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_upload_file(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_file = MagicMock(spec=FileObject)
        mock_file.id = "file-123"
        mock_client.files.create = AsyncMock(return_value=mock_file)
        mock_client.files.wait_for_processing = AsyncMock(return_value=mock_file)

        llm = LLM(api_key="test")
        result = await llm.upload_file("file_content", "training")

        assert result == mock_file
        mock_client.files.create.assert_called_once()
        mock_client.files.wait_for_processing.assert_called_once_with("file-123")

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_fine_tune_default_params(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_job = MagicMock()
        mock_client.fine_tuning.jobs.create = AsyncMock(return_value=mock_job)

        mock_file1 = MagicMock(id="file-1")
        mock_file2 = MagicMock(id="file-2")

        llm = LLM(api_key="test")
        result = await llm.fine_tune(mock_file1, mock_file2)

        assert result == mock_job
        mock_client.fine_tuning.jobs.create.assert_called_once()

        call_kwargs = mock_client.fine_tuning.jobs.create.call_args[1]
        assert call_kwargs["training_file"] == "file-1"
        assert call_kwargs["validation_file"] == "file-2"
        assert call_kwargs["model"] == "gpt-4.1-nano"

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_fine_tune_custom_params(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_job = MagicMock()
        mock_client.fine_tuning.jobs.create = AsyncMock(return_value=mock_job)

        mock_file1 = MagicMock(id="file-1")
        mock_file2 = MagicMock(id="file-2")

        llm = LLM(api_key="test")
        result = await llm.fine_tune(
            mock_file1,
            mock_file2,
            suffix="custom",
            model="gpt-4",
            n_epochs=3,
            batch_size=8,
            learning_rate_multiplier=0.5,
        )

        assert result == mock_job

        call_kwargs = mock_client.fine_tuning.jobs.create.call_args[1]
        assert call_kwargs["suffix"] == "custom"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["hyperparameters"]["n_epochs"] == 3
        assert call_kwargs["hyperparameters"]["batch_size"] == 8
        assert call_kwargs["hyperparameters"]["learning_rate_multiplier"] == 0.5

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_wait_for_fine_tune(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_job = MagicMock()
        mock_job.finished_at = "2024-01-01T00:00:00"

        mock_client.fine_tuning.jobs.retrieve = AsyncMock(return_value=mock_job)

        llm = LLM(api_key="test")
        result = await llm.wait_for_fine_tune("job-123", poll_time=1)

        assert result == mock_job
        mock_client.fine_tuning.jobs.retrieve.assert_called_once()

    @patch("friday.core.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_wait_for_fine_tune_polls(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        job_in_progress = MagicMock(finished_at=None)
        job_completed = MagicMock(finished_at="2024-01-01T00:00:00")

        mock_client.fine_tuning.jobs.retrieve = AsyncMock(
            side_effect=[job_in_progress, job_completed]
        )

        llm = LLM(api_key="test")
        result = await llm.wait_for_fine_tune("job-123", poll_time=1)

        assert result == job_completed
        assert mock_client.fine_tuning.jobs.retrieve.call_count == 2

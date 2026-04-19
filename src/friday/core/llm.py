import asyncio
import os
from typing import (
    Any,
    AsyncGenerator,
    Mapping,
    Type,
    TypeVar,
    TypedDict,
    Union,
    Unpack,
)
from dotenv import load_dotenv
import httpx
from openai.types.responses import (
    Response,
    ParsedResponse,
)
from openai import (
    DEFAULT_MAX_RETRIES,
    NOT_GIVEN,
    NotGiven,
    AsyncOpenAI,
)
from openai.types.file_object import FileObject
from friday.core.tool_shed import ToolShed


load_dotenv()


T = TypeVar("T")


class LLMOpenAISDKConstructorArgs(TypedDict, total=False):
    organization: str | None = (None,)
    project: str | None = (None,)
    webhook_secret: str | None = (None,)
    base_url: str | httpx.URL | None = (None,)
    websocket_base_url: str | httpx.URL | None = (None,)
    timeout: float | httpx.Timeout | None | NotGiven = (NOT_GIVEN,)
    max_retries: int = (DEFAULT_MAX_RETRIES,)
    default_headers: Mapping[str, str] | None = (None,)
    default_query: Mapping[str, object] | None = (None,)


class LLM:
    def __init__(
        self,
        api_key: str = None,
        model="gpt-4.1-nano",
        embedding_model="text-embedding-3-small",
        tool_shed: ToolShed = ToolShed(),
        **kwargs: Unpack[LLMOpenAISDKConstructorArgs],
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.tool_shed = tool_shed
        self.client = AsyncOpenAI(
            api_key=(api_key if api_key else os.environ.get("OPENAI_API_KEY")), **kwargs
        )

    async def invoke(
        self, input: list[dict], response_format: Type[T] = str
    ) -> Union[Response, ParsedResponse[Type[T]]]:
        if response_format == str:
            response = await self.client.responses.create(
                model=self.model,
                tools=self.tool_shed.definitions if self.tool_shed else [],
                input=input,
            )
            return response
        else:
            response = await self.client.responses.parse(
                model=self.model,
                tools=self.tool_shed.definitions if self.tool_shed else [],
                input=input,
                text_format=response_format,
            )
            return response

    async def stream(
        self, input: list[dict]
    ) -> AsyncGenerator[tuple[dict, list[dict]]]:
        message: dict = {}
        async with self.client.responses.stream(
            model=self.model,
            tools=self.tool_shed.definitions if self.tool_shed else [],
            input=input,
        ) as stream:
            if stream:
                async for event in stream:
                    if event.type == "response.created":
                        pass
                    elif event.type == "response.in_progress":
                        pass
                    elif event.type == "response.completed":
                        message = {
                            **message,
                            "status": event.response.output[0].status,
                            "content": "",
                        }
                        yield message
                    elif event.type == "response.failed":
                        pass
                    elif event.type == "response.output_item.added":
                        entry = event.item.model_dump()
                        if "content" in entry:
                            message = {**message, **entry, "content": ""}
                        else:
                            message = {**message, **entry}
                        yield message
                    elif event.type == "response.output_item.done":
                        pass
                    elif event.type == "response.content_part.added":
                        pass
                    elif event.type == "response.content_part.done":
                        pass
                    elif event.type == "response.output_text.delta":
                        message = {**message, "content": event.delta}
                        yield message
                    elif event.type == "response.output_text.done":
                        pass
                    elif event.type == "response.output_text.annotation_added":
                        pass
                    elif event.type == "response.text.done":
                        pass
                    elif event.type == "response.function_call_arguments.delta":
                        message = {
                            **message,
                            "arguments": message["arguments"] + event.delta,
                        }
                    elif event.type == "response.function_call_arguments.done":
                        pass
                    elif event.type == "response.refusal.delta":
                        pass
                    elif event.type == "response.refusal.done":
                        pass
                    elif event.type == "response.file_search_call.in_progress":
                        pass
                    elif event.type == "response.file_search_call.searching":
                        pass
                    elif event.type == "response.file_search_call.completed":
                        pass
                    elif event.type == "error":
                        raise Exception(f"an unexpected error occurred")
                    else:
                        raise Exception(f"unsupported type {event.type}")

    async def embedding(self, input: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.embedding_model, input=input
        )
        return response.data[0].embedding

    async def embeddings(self, input: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.embedding_model, input=input
        )
        return [e.embedding for e in response.data]

    async def upload_file(self, file: Any, purpose: str):
        file_object = await self.client.files.create(file=file, purpose=purpose)
        file_object = await self.client.files.wait_for_processing(file_object.id)
        return file_object

    async def fine_tune(
        self,
        train_file_object: FileObject,
        validate_file_object: FileObject,
        suffix=None,
        model: str = None,
        n_epochs: int = 1,
        batch_size: int = 1,
        learning_rate_multiplier: float = 0.1,
    ):
        local_modal = model if model is not None else self.model
        job = await self.client.fine_tuning.jobs.create(
            training_file=train_file_object.id,
            validation_file=validate_file_object.id,
            model=local_modal,
            hyperparameters={
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate_multiplier,
            },
            suffix=suffix,
        )
        return job

    async def wait_for_fine_tune(self, job_id: str, poll_time: int = 5):
        while True:
            job = await self.client.fine_tuning.jobs.retrieve(job_id)
            if job.finished_at is not None:
                return job
            await asyncio.sleep(poll_time)

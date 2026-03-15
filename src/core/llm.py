import asyncio
import os
from typing import Any, Type, TypeVar
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.file_object import FileObject


load_dotenv()


T = TypeVar("T")


class LLM:
    def __init__(
        self,
        openai_api_key=None,
        model="gpt-4.1-nano",
        embedding_model="text-embedding-3-small",
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.client = OpenAI(
            api_key=(
                openai_api_key if openai_api_key else os.environ.get("OPENAI_API_KEY")
            )
        )

    def invoke(self, input: list[Any], response_format: Type[T] = str) -> T:
        if response_format == str:
            response = self.client.responses.create(model=self.model, input=input)
            return response.output_text
        else:
            response = self.client.responses.parse(
                model=self.model, input=input, text_format=response_format
            )
            return response.output_parsed

    def stream(self, input: list[Any], tools: list[Any]):
        return self.client.responses.stream(
            model=self.model,
            tools=tools,
            input=input,
        )

    def embedding(self, input: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model, input=input
        )
        return response.data[0].embedding

    def embeddings(self, input: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model, input=input
        )
        return [e.embedding for e in response.data]

    def upload_file(self, file: Any, purpose: str):
        file_object = self.client.files.create(file=file, purpose=purpose)
        file_object = self.client.files.wait_for_processing(file_object.id)
        return file_object

    def fine_tune(
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
        job = self.client.fine_tuning.jobs.create(
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
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            if job.finished_at is not None:
                return job
            await asyncio.sleep(poll_time)

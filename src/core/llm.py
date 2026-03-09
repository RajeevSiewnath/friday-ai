import os
from typing import Any, Type, TypeVar
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


T = TypeVar("T")


class LLM:
    client: OpenAI
    model: str
    embedding_model: str

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

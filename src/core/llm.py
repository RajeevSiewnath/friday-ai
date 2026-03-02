import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def invoke(input: list[Any], response_format: Any = None) -> str | Any:
    if response_format:
        response = client.responses.parse(
            model="gpt-4.1-nano", input=input, text_format=response_format
        )
        return response.output_parsed
    else:
        response = client.responses.create(model="gpt-4.1-nano", input=input)
        return response.output_text


def embedding(input: str) -> list[float]:
    response = client.embeddings.create(model="text-embedding-3-small", input=input)
    return response.data[0].embedding

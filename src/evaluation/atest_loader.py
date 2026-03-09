import json
import os
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Any

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
chroma = chromadb.Client(Settings(is_persistent=True))


class Document(TypedDict):
    id: str
    document: str
    metadata: dict[str, Any]


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(
        description="Keywords that must appear in retrieved context"
    )
    answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category")


def fetch_context(question: str) -> list[Document]:
    """Fetch context of a question from the vector db"""
    response = client.embeddings.create(model="text-embedding-3-small", input=question)
    embedding = response.data[0].embedding

    collection = chroma.get_collection(name="cv-rajeev-siewnath")
    results = collection.query(query_embeddings=embedding, n_results=10)

    return [
        Document(id=id, document=document, metadata=metadata)
        for id, document, metadata in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]


def invoke_llm(input: list[Any], response_format: Any = None) -> str | Any:
    if response_format:
        response = client.responses.parse(
            model="gpt-4.1-nano", input=input, text_format=response_format
        )
        return response.output_parsed
    else:
        response = client.responses.create(model="gpt-4.1-nano", input=input)
        return response.output_text


def load_tests(file: str) -> list[TestQuestion]:
    """Load test questions from JSON file."""
    tests = []
    with open(file, "r", encoding="utf-8") as f:
        tests = [TestQuestion(**question) for question in json.loads(f.read())]
    return tests

from typing import Any, Union
from pydantic import BaseModel, Field


class JsonDocument(BaseModel):
    """A document class that represents an JSON document."""

    id: str = Field(description="The id of the vectorizable JSON")
    document: str = Field(description="The document entry of the vectorizable JSON")
    type: str = Field(description="The document type")
    source: str = Field(description="The document path")
    metadata: dict[str, Any] = Field(
        description="The document entry of the vectorizable JSON"
    )

    def __add__(
        self, other: Union["JsonDocumentCollection", "JsonDocument"]
    ) -> "JsonDocumentCollection":
        if isinstance(other, JsonDocumentCollection):
            return JsonDocumentCollection(
                json_documents=self.json_documents + other.json_documents
            )
        elif isinstance(other, JsonDocument):
            return JsonDocumentCollection(json_documents=self.json_documents + [other])
        else:
            raise Exception(f"cannot add {other} to JsonDocumentCollection")


class JsonDocumentCollection(BaseModel):
    """A document collection"""

    json_documents: list[JsonDocument] = Field(
        description="A list of JsonDocument instances", default=[]
    )

    @classmethod
    def from_docs(cls, docs: list[JsonDocument]) -> "JsonDocumentCollection":
        collection = cls()
        collection.json_documents = docs
        return collection

    def __add__(
        self, other: Union["JsonDocumentCollection", JsonDocument]
    ) -> "JsonDocumentCollection":
        if isinstance(other, JsonDocumentCollection):
            return JsonDocumentCollection(
                json_documents=self.json_documents + other.json_documents
            )
        elif isinstance(other, JsonDocument):
            return JsonDocumentCollection(json_documents=self.json_documents + [other])
        else:
            raise Exception(f"cannot add {other} to JsonDocumentCollection")

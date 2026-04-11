from copy import deepcopy
from typing import Any, Optional, Union
from pydantic import BaseModel, Field


class Document(BaseModel):
    path: Optional[str] = Field(description="The file path")
    type: Optional[str] = Field(description="The type of file based on the path")
    id: str = Field(description="The document id")
    content: str = Field(description="The content field")
    metadata: dict[str, Any] = Field(description="The metadata of the document")
    metrics: list[Any] = Field(
        description="A metrics list for debugging", default_factory=list
    )

    def __add__(
        self, other: Union["DocumentCollection", "Document"]
    ) -> "DocumentCollection":
        if isinstance(other, DocumentCollection):
            return DocumentCollection(
                documents=[deepcopy(self)] + deepcopy(other.documents)
            )
        elif isinstance(other, Document):
            return DocumentCollection(documents=[deepcopy(self), deepcopy(other)])
        else:
            raise Exception(f"cannot add {other} to DocumentCollection")


class DocumentCollection(BaseModel):
    documents: list[Document] = Field(
        description="A list of Document instances", default=[]
    )

    @classmethod
    def from_docs(cls, docs: list[Document]) -> "DocumentCollection":
        collection = cls()
        collection.documents = docs
        return collection

    def __add__(
        self, other: Union["DocumentCollection", Document]
    ) -> "DocumentCollection":
        if isinstance(other, DocumentCollection):
            return DocumentCollection(
                documents=deepcopy(self.documents) + deepcopy(other.documents)
            )
        elif isinstance(other, Document):
            return DocumentCollection(
                documents=deepcopy(self.documents) + [deepcopy(other)]
            )
        else:
            raise Exception(f"cannot add {other} to DocumentCollection")

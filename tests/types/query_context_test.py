from copy import deepcopy
from typing import Any, Union
from pydantic import BaseModel, Field


class RagContext(BaseModel):
    id: str = Field(description="The document id")
    content: str = Field(description="The content field")
    metadata: dict[str, Any] = Field(
        description="The metadata of the document", default_factory=dict
    )

    def __add__(
        self, other: Union["RagContextCollection", "RagContext"]
    ) -> "RagContextCollection":
        if isinstance(other, RagContextCollection):
            return RagContextCollection(
                contexts=[deepcopy(self)] + deepcopy(other.contexts)
            )
        elif isinstance(other, RagContext):
            return RagContextCollection(contexts=[deepcopy(self), deepcopy(other)])
        else:
            raise Exception(f"cannot add {other} to RagContextCollection")


class RagContextCollection(BaseModel):
    contexts: list[RagContext] = Field(
        description="A list of RagContext instances", default=[]
    )

    @classmethod
    def from_contexts(cls, docs: list[RagContext]) -> "RagContextCollection":
        collection = cls()
        collection.contexts = docs
        return collection

    def __add__(
        self, other: Union["RagContextCollection", RagContext]
    ) -> "RagContextCollection":
        if isinstance(other, RagContextCollection):
            return RagContextCollection(
                contexts=deepcopy(self.contexts) + deepcopy(other.contexts)
            )
        elif isinstance(other, RagContext):
            return RagContextCollection(
                contexts=deepcopy(self.contexts) + [deepcopy(other)]
            )
        else:
            raise Exception(f"cannot add {other} to RagContextCollection")


class QueryContext(BaseModel):
    @property
    def question(self):
        return self.question_history[-1]

    @question.setter
    def question(self, value: str):
        self.question_history.append(value)

    question_history: list[str] = Field(
        description="The history of the question, 0 being the first",
        default_factory=list,
    )

    history: list[Any] = Field(description="The chat history")
    context: RagContextCollection = Field(
        default_factory=RagContextCollection,
        description="The context loaded in by RAG vectorized databases",
    )

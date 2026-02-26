from typing import Any
from pydantic import BaseModel, Field

from invocation.JsonDocument import JsonDocument


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document: JsonDocument):
        metadata = {
            **document.metadata,
            "source": document.source,
            "type": document.type,
        }
        return ChunkResult(
            id=document.id,
            metadata=metadata,
            document=self.headline
            + "\n\n"
            + self.summary
            + "\n\n"
            + self.original_text,
        )


class ChunkResult(BaseModel):
    id: str = Field(description="The id of the vectorizable JSON")
    document: str = Field(description="The document entry of the vectorizable JSON")
    metadata: dict[str, Any] = Field(
        description="The document entry of the vectorizable JSON"
    )


class Chunks(BaseModel):
    chunks: list[Chunk] = Field(description="Collection of chunks")

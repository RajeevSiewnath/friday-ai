from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    path: Optional[str] = Field(description="The file path")
    type: Optional[str] = Field(description="The type of file based on the path")
    id: str = Field(description="The document id")
    content: str = Field(description="The content field")
    metadata: dict[str, Any] = Field(description="The metadata of the document")

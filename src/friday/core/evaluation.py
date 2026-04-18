from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(
        description="Keywords that must appear in retrieved context",
        default_factory=list,
    )
    answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category")

from copy import deepcopy
from typing import Any, Union
from pydantic import BaseModel, Field
from evaluation.test_loader import Document
from pipelines.abstract_pipeline import AbstractPipeline


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(
        description="Keywords that must appear in retrieved context",
        default_factory=list,
    )
    answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category")

    def __add__(
        self, other: Union["TestQuestionCollection", "TestQuestion"]
    ) -> "TestQuestionCollection":
        if isinstance(other, TestQuestionCollection):
            return TestQuestionCollection(
                questions=[deepcopy(self)] + deepcopy(other.questions)
            )
        elif isinstance(other, TestQuestion):
            return TestQuestionCollection(questions=[deepcopy(self), deepcopy(other)])
        else:
            raise Exception(f"cannot add {other} to TestQuestionCollection")


class TestQuestionCollection(BaseModel):
    questions: list[TestQuestion] = Field(
        description="A list of TestQuestion instances", default=[]
    )

    @classmethod
    def from_questions(cls, docs: list[TestQuestion]) -> "TestQuestionCollection":
        collection = cls()
        collection.questions = docs
        return collection

    def __add__(
        self, other: Union["TestQuestionCollection", TestQuestion]
    ) -> "TestQuestionCollection":
        if isinstance(other, TestQuestionCollection):
            return TestQuestionCollection(
                questions=deepcopy(self.questions) + deepcopy(other.questions)
            )
        elif isinstance(other, TestQuestion):
            return TestQuestionCollection(
                questions=deepcopy(self.questions) + [deepcopy(other)]
            )
        else:
            raise Exception(f"cannot add {other} to TestQuestionCollection")


class EvaluationScore(BaseModel):
    questions: TestQuestionCollection = Field(
        description="The list of TestQuestions", default_factory=TestQuestionCollection
    )
    scores: list[Any] = Field(
        description="The list for scores to keep", default_factory=list
    )


class OptimizationsPipeline(AbstractPipeline[EvaluationScore]):
    pass

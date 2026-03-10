from copy import deepcopy
from typing import Any, Union
from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(
        description="Keywords that must appear in retrieved context",
        default_factory=list,
    )
    answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category")

    def __add__(
        self, other: Union["EvalQuestionCollection", "EvalQuestion"]
    ) -> "EvalQuestionCollection":
        if isinstance(other, EvalQuestionCollection):
            return EvalQuestionCollection(
                questions=[deepcopy(self)] + deepcopy(other.questions)
            )
        elif isinstance(other, EvalQuestion):
            return EvalQuestionCollection(questions=[deepcopy(self), deepcopy(other)])
        else:
            raise Exception(f"cannot add {other} to TestQuestionCollection")


class EvalQuestionCollection(BaseModel):
    questions: list[EvalQuestion] = Field(
        description="A list of TestQuestion instances", default=[]
    )

    @classmethod
    def from_questions(cls, docs: list[EvalQuestion]) -> "EvalQuestionCollection":
        collection = cls()
        collection.questions = docs
        return collection

    def __add__(
        self, other: Union["EvalQuestionCollection", EvalQuestion]
    ) -> "EvalQuestionCollection":
        if isinstance(other, EvalQuestionCollection):
            return EvalQuestionCollection(
                questions=deepcopy(self.questions) + deepcopy(other.questions)
            )
        elif isinstance(other, EvalQuestion):
            return EvalQuestionCollection(
                questions=deepcopy(self.questions) + [deepcopy(other)]
            )
        else:
            raise Exception(f"cannot add {other} to TestQuestionCollection")


class EvaluationScore(BaseModel):
    questions: EvalQuestionCollection = Field(
        description="The list of TestQuestions", default_factory=EvalQuestionCollection
    )
    scores: list[Any] = Field(
        description="The list for scores to keep", default_factory=list
    )

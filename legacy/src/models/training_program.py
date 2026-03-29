from copy import deepcopy
from typing import Union
from pydantic import BaseModel, Field


class TrainingQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(
        description="Keywords that must appear in retrieved context",
        default_factory=list,
    )
    answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category")

    def __add__(
        self, other: Union["TrainingQuestionCollection", "TrainingQuestion"]
    ) -> "TrainingQuestionCollection":
        if isinstance(other, TrainingQuestionCollection):
            return TrainingQuestionCollection(
                questions=[deepcopy(self)] + deepcopy(other.questions)
            )
        elif isinstance(other, TrainingQuestion):
            return TrainingQuestionCollection(
                questions=[deepcopy(self), deepcopy(other)]
            )
        else:
            raise Exception(f"cannot add {other} to TestQuestionCollection")


class TrainingQuestionCollection(BaseModel):
    questions: list[TrainingQuestion] = Field(
        description="A list of TestQuestion instances", default=[]
    )

    @classmethod
    def from_questions(
        cls, docs: list[TrainingQuestion]
    ) -> "TrainingQuestionCollection":
        collection = cls()
        collection.questions = docs
        return collection

    def __add__(
        self, other: Union["TrainingQuestionCollection", TrainingQuestion]
    ) -> "TrainingQuestionCollection":
        if isinstance(other, TrainingQuestionCollection):
            return TrainingQuestionCollection(
                questions=deepcopy(self.questions) + deepcopy(other.questions)
            )
        elif isinstance(other, TrainingQuestion):
            return TrainingQuestionCollection(
                questions=deepcopy(self.questions) + [deepcopy(other)]
            )
        else:
            raise Exception(f"cannot add {other} to TestQuestionCollection")


class TrainingProgram(BaseModel):
    full: TrainingQuestionCollection = Field(
        description="The full list of test questions",
        default_factory=TrainingQuestionCollection,
    )
    training: TrainingQuestionCollection = Field(
        description="The training list of test questions",
        default_factory=TrainingQuestionCollection,
    )
    validating: TrainingQuestionCollection = Field(
        description="The validation list of test questions",
        default_factory=TrainingQuestionCollection,
    )
    testing: TrainingQuestionCollection = Field(
        description="The test list of test questions",
        default_factory=TrainingQuestionCollection,
    )
    statistics: dict[str, float] = Field(
        description="The statistics of the training", default_factory=dict
    )

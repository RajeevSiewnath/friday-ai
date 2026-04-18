from operator import add
from typing import Annotated, TypedDict
from friday.core.evaluation import Evaluation


class QuestionsState(TypedDict):
    evaluation_questions: Annotated[list[Evaluation], add]

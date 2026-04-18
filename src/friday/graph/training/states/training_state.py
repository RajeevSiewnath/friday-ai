from typing import TypedDict
from friday.core.evaluation import Evaluation


class TrainingState(TypedDict):
    full_data_set: list[Evaluation]
    training_data_set: list[Evaluation]
    validating_data_set: list[Evaluation]
    testing_data_set: list[Evaluation]

from copy import deepcopy
import json
import math
from pathlib import Path
import random
from tqdm import tqdm
from pipelines.pipeline import Pipe
from models.training_program import (
    TrainingQuestion,
    TrainingQuestionCollection,
    TrainingProgram,
)


class DataSetsLoader(Pipe[TrainingProgram]):
    def __init__(
        self, path: str, max=None, training_ratio=8, validating_ratio=1, testing_ratio=1
    ):
        super().__init__()
        self.path = path
        self.max = max
        self.training_ratio = training_ratio
        self.validating_ratio = validating_ratio
        self.testing_ratio = testing_ratio

    def run(self, input):
        file_path = Path(self.path)
        tests = []
        with open(file_path, "r", encoding="utf-8") as f:
            files = (
                json.loads(f.read())
                if self.max is None
                else json.loads(f.read())[: self.max]
            )
            tests = [TrainingQuestion(**question) for question in tqdm(files)]
        input.full = TrainingQuestionCollection.from_questions(tests)

        questions = deepcopy(input.full.questions)
        random.shuffle(questions)

        total_ratio = self.training_ratio + self.validating_ratio + self.testing_ratio
        total_questions = len(questions)
        training_set = math.floor((self.training_ratio / total_ratio) * total_questions)
        validating_set = math.floor(
            (self.validating_ratio / total_ratio) * total_questions
        )

        input.training = TrainingQuestionCollection.from_questions(
            questions[:training_set]
        )
        input.validating = TrainingQuestionCollection.from_questions(
            questions[training_set : training_set + validating_set]
        )
        input.testing = TrainingQuestionCollection.from_questions(
            questions[training_set + validating_set :]
        )

        return input

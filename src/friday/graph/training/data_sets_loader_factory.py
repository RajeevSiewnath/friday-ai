from copy import deepcopy
import json
import math
from pathlib import Path
import random
from tqdm import tqdm
from friday.core.evaluation import Evaluation


def data_sets_loader_factory(
    path: str, max=None, training_ratio=8, validating_ratio=1, testing_ratio=1
):
    def data_sets_loader():
        file_path = Path(path)
        with open(file_path, "r", encoding="utf-8") as f:
            questions = (
                json.loads(f.read()) if max is None else json.loads(f.read())[:max]
            )
        full = [Evaluation(**question) for question in tqdm(questions)]

        questions = deepcopy(full)
        random.shuffle(questions)

        total_ratio = training_ratio + validating_ratio + testing_ratio
        total_questions = len(questions)
        training_set = math.floor((training_ratio / total_ratio) * total_questions)
        validating_set = math.floor((validating_ratio / total_ratio) * total_questions)

        training = questions[:training_set]
        validating = questions[training_set : training_set + validating_set]
        testing = questions[training_set + validating_set :]

        return {
            "full_data_set": full,
            "training_data_set": training,
            "validating_data_set": validating,
            "testing_data_set": testing,
        }

    return data_sets_loader

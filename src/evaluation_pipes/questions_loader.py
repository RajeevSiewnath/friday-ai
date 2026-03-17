import json
from pathlib import Path
from tqdm import tqdm
from pipelines.pipeline import Pipe
from models.evaluation_score import (
    EvalQuestion,
    EvalQuestionCollection,
    EvaluationScore,
)


class QuestionsLoader(Pipe[EvaluationScore]):
    def __init__(self, path: str, max=None):
        super().__init__()
        self.path = path
        self.max = max

    def run(self, input):
        file_path = Path(self.path)
        tests = []
        with open(file_path, "r", encoding="utf-8") as f:
            files = (
                json.loads(f.read())
                if self.max is None
                else json.loads(f.read())[: self.max]
            )
            tests = [EvalQuestion(**question) for question in tqdm(files)]
        input.questions = EvalQuestionCollection.from_questions(tests)
        return input

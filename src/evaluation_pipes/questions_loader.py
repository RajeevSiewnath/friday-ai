import json
from pathlib import Path
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.evaluation_pipeline import (
    EvaluationScore,
    TestQuestion,
    TestQuestionCollection,
)


class QuestionsLoader(AbstractPipe[EvaluationScore]):
    path: str

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def pipe(self, input):
        file_path = Path(self.path)
        tests = []
        with open(file_path, "r", encoding="utf-8") as f:
            tests = [TestQuestion(**question) for question in json.loads(f.read())]
        input.questions = TestQuestionCollection.from_questions(tests)
        return input


if __name__ == "__main__":
    eval_score = EvaluationScore()
    print(QuestionsLoader("rag_evaluation.json").pipe(eval_score))

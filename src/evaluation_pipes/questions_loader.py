import json
from pathlib import Path
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.evaluation_pipeline import (
    EvaluationScore,
    EvalQuestion,
    EvalQuestionCollection,
)


class QuestionsLoader(AbstractPipe[EvaluationScore]):
    path: str

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def pipe(self, input, prompt_context, llm):
        file_path = Path(self.path)
        tests = []
        with open(file_path, "r", encoding="utf-8") as f:
            tests = [EvalQuestion(**question) for question in json.loads(f.read())]
        input.questions = EvalQuestionCollection.from_questions(tests)
        return input


if __name__ == "__main__":
    eval_score = EvaluationScore()
    print(QuestionsLoader("rag_evaluation.json").pipe(eval_score))

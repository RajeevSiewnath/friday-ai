import json
from pathlib import Path
from friday.core.evaluation import Evaluation
from friday.graph.evaluation.states.questions_state import QuestionsState


def questions_loader_factory(path: str, max=None):
    def questions_loader(state: QuestionsState) -> QuestionsState:
        file_path = Path(path)
        with open(file_path, "r", encoding="utf-8") as f:
            questions = (
                json.loads(f.read()) if max is None else json.loads(f.read())[:max]
            )
        return {
            "evaluation_questions": [Evaluation(**question) for question in questions]
        }

    return questions_loader

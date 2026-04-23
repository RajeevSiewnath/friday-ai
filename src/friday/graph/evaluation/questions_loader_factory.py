import json
from pathlib import Path
from friday.core.evaluation import Evaluation
from friday.graph.evaluation.states.questions_state import QuestionsState
from friday.loggers.logger import Logger


def questions_loader_factory(path: str, max=None):
    def questions_loader(state: QuestionsState) -> QuestionsState:
        logger = Logger.get_logger("node.questions_loader")
        logger.info("loading questions from path %s", path)

        file_path = Path(path)
        with open(file_path, "r", encoding="utf-8") as f:
            questions = (
                json.loads(f.read()) if max is None else json.loads(f.read())[:max]
            )

        logger.debug(
            "questions: %s", lambda: [Evaluation(**question) for question in questions]
        )
        return {
            "evaluation_questions": [Evaluation(**question) for question in questions]
        }

    return questions_loader

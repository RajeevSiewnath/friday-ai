import pytest
import tempfile
import json
import os
from friday.core.evaluation import Evaluation
from friday.graph.evaluation.questions_loader_factory import questions_loader_factory
from friday.graph.evaluation.states.questions_state import QuestionsState


class TestQuestionsLoaderFactory:
    def test_loader_factory_returns_callable(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        assert callable(loader)

    def test_loader_returns_questions_state(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)
        assert isinstance(result, dict)
        assert "evaluation_questions" in result

    def test_loader_returns_evaluation_objects(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)
        assert all(isinstance(q, Evaluation) for q in result["evaluation_questions"])

    def test_loader_respects_max_parameter(self, evaluation_json_file):
        max_items = 1
        loader = questions_loader_factory(evaluation_json_file, max=max_items)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)
        assert len(result["evaluation_questions"]) <= max_items

    def test_loader_without_max_loads_all(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)
        assert len(result["evaluation_questions"]) == 3

    def test_loader_extracts_evaluation_properties(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)

        for question in result["evaluation_questions"]:
            assert hasattr(question, "question")
            assert hasattr(question, "answer")
            assert hasattr(question, "category")
            assert hasattr(question, "keywords")

    def test_loader_with_custom_path(self, sample_evaluations):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_eval.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in sample_evaluations], f)

            loader = questions_loader_factory(file_path)
            state = QuestionsState(evaluation_questions=[])
            result = loader(state)
            assert len(result["evaluation_questions"]) == len(sample_evaluations)

    def test_loader_preserves_question_content(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)

        assert len(result["evaluation_questions"]) > 0
        first_q = result["evaluation_questions"][0]
        assert len(first_q.question) > 0

    def test_loader_preserves_answer_content(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)

        for question in result["evaluation_questions"]:
            assert isinstance(question.answer, str)

    def test_loader_preserves_category(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)

        for question in result["evaluation_questions"]:
            assert isinstance(question.category, str)

    def test_loader_creates_valid_evaluations(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)

        for question in result["evaluation_questions"]:
            assert Evaluation(**question.model_dump())

    def test_loader_max_zero(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file, max=0)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)
        assert len(result["evaluation_questions"]) == 0

    def test_loader_max_exceeds_available(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file, max=1000)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)
        assert len(result["evaluation_questions"]) == 3

    def test_loader_callable_multiple_times(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])

        result1 = loader(state)
        result2 = loader(state)

        assert len(result1["evaluation_questions"]) == len(
            result2["evaluation_questions"]
        )

    def test_loader_with_keywords(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state = QuestionsState(evaluation_questions=[])
        result = loader(state)

        for question in result["evaluation_questions"]:
            assert isinstance(question.keywords, list)

    def test_loader_returns_new_state_dict(self, evaluation_json_file):
        loader = questions_loader_factory(evaluation_json_file)
        state1 = QuestionsState(evaluation_questions=[])
        state2 = QuestionsState(evaluation_questions=[])

        result1 = loader(state1)
        result2 = loader(state2)

        assert result1 is not result2

import pytest
from friday.core.evaluation import Evaluation
from friday.graph.evaluation.states.questions_state import QuestionsState


class TestQuestionsState:
    def test_questions_state_creation(self, questions_state: QuestionsState):
        assert "evaluation_questions" in questions_state
        assert isinstance(questions_state["evaluation_questions"], list)

    def test_questions_state_has_evaluation_questions_key(
        self, questions_state: QuestionsState
    ):
        assert "evaluation_questions" in questions_state

    def test_questions_state_is_list(self, questions_state: QuestionsState):
        assert isinstance(questions_state["evaluation_questions"], list)

    def test_questions_state_contains_evaluations(
        self, questions_state: QuestionsState
    ):
        assert all(
            isinstance(q, Evaluation) for q in questions_state["evaluation_questions"]
        )

    def test_questions_state_empty(self, empty_questions_state: QuestionsState):
        assert empty_questions_state["evaluation_questions"] == []

    def test_questions_state_annotations(self):
        assert hasattr(QuestionsState, "__annotations__")
        assert "evaluation_questions" in QuestionsState.__annotations__

    def test_questions_state_multiple_questions(
        self, questions_state: QuestionsState
    ):
        assert len(questions_state["evaluation_questions"]) > 0
        for q in questions_state["evaluation_questions"]:
            assert hasattr(q, "question")
            assert hasattr(q, "answer")
            assert hasattr(q, "category")
            assert hasattr(q, "keywords")

    def test_questions_state_evaluation_attributes(
        self, questions_state: QuestionsState
    ):
        evaluation = questions_state["evaluation_questions"][0]
        assert isinstance(evaluation.question, str)
        assert isinstance(evaluation.answer, str)
        assert isinstance(evaluation.keywords, list)
        assert isinstance(evaluation.category, str)

    def test_questions_state_can_be_modified(
        self, questions_state: QuestionsState
    ):
        original_count = len(questions_state["evaluation_questions"])
        new_eval = Evaluation(
            question="New question?",
            answer="New answer",
            category="test",
            keywords=["test"],
        )
        questions_state["evaluation_questions"].append(new_eval)
        assert len(questions_state["evaluation_questions"]) == original_count + 1

    def test_questions_state_with_keywords(self, questions_state: QuestionsState):
        for q in questions_state["evaluation_questions"]:
            assert isinstance(q.keywords, list)
            if q.keywords:
                for keyword in q.keywords:
                    assert isinstance(keyword, str)

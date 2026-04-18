import pytest
from friday.graph.evaluation.answer_eval_factory import (
    AnswerEvalResult,
    AnswerEvalState,
    AnswerEvalContext,
    answer_eval_factory,
)


class TestAnswerEvalResult:
    def test_answer_eval_result_creation(self):
        result = AnswerEvalResult(
            feedback="Good answer",
            accuracy=4.5,
            completeness=4.0,
            relevance=4.8,
        )
        assert result.feedback == "Good answer"
        assert result.accuracy == 4.5
        assert result.completeness == 4.0
        assert result.relevance == 4.8

    def test_answer_eval_result_all_fields_required(self):
        with pytest.raises(Exception):
            AnswerEvalResult(feedback="Good", accuracy=4.0)

    def test_accuracy_percentage(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=5, completeness=5, relevance=5
        )
        assert result.accuracy_percentage == 1.0

    def test_accuracy_percentage_minimum(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=1, completeness=5, relevance=5
        )
        assert result.accuracy_percentage == 0.0

    def test_accuracy_percentage_middle(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=3, completeness=5, relevance=5
        )
        assert result.accuracy_percentage == 0.5

    def test_completeness_percentage(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=5, completeness=5, relevance=5
        )
        assert result.completeness_percentage == 1.0

    def test_completeness_percentage_minimum(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=5, completeness=1, relevance=5
        )
        assert result.completeness_percentage == 0.0

    def test_relevance_percentage(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=5, completeness=5, relevance=5
        )
        assert result.relevance_percentage == 1.0

    def test_relevance_percentage_minimum(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=5, completeness=5, relevance=1
        )
        assert result.relevance_percentage == 0.0

    def test_score_ranges(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=2.5, completeness=3.5, relevance=4.2
        )
        assert 0 <= result.accuracy_percentage <= 1
        assert 0 <= result.completeness_percentage <= 1
        assert 0 <= result.relevance_percentage <= 1

    def test_result_with_float_scores(self):
        result = AnswerEvalResult(
            feedback="Detailed feedback",
            accuracy=4.2,
            completeness=3.8,
            relevance=4.5,
        )
        assert result.accuracy == 4.2
        assert result.completeness == 3.8
        assert result.relevance == 4.5


class TestAnswerEvalFactory:
    def test_factory_returns_callable(self):
        factory = answer_eval_factory(
            score_key="test_score",
            collection_key="test_collection",
            user_context="Test context",
        )
        assert callable(factory)

    def test_factory_with_default_retrieval_k(self):
        factory = answer_eval_factory(
            score_key="test",
            collection_key="collection",
            user_context="context",
        )
        assert callable(factory)

    def test_factory_with_custom_retrieval_k(self):
        factory = answer_eval_factory(
            score_key="test",
            collection_key="collection",
            user_context="context",
            retrieval_k=5,
        )
        assert callable(factory)

    def test_factory_with_large_retrieval_k(self):
        factory = answer_eval_factory(
            score_key="test",
            collection_key="collection",
            user_context="context",
            retrieval_k=100,
        )
        assert callable(factory)

    def test_answer_eval_state_creation(self, questions_state, scores_state):
        state = AnswerEvalState(
            evaluation_questions=questions_state["evaluation_questions"],
            evaluation_scores=scores_state["evaluation_scores"],
        )
        assert "evaluation_questions" in state
        assert "evaluation_scores" in state

    def test_answer_eval_state_with_empty_scores(self, questions_state):
        state = AnswerEvalState(
            evaluation_questions=questions_state["evaluation_questions"],
            evaluation_scores={},
        )
        assert state["evaluation_scores"] == {}

    def test_answer_eval_state_inheritance(self, questions_state, scores_state):
        from friday.graph.evaluation.states.questions_state import QuestionsState
        from friday.graph.evaluation.states.scores_state import ScoresState

        state = AnswerEvalState(
            evaluation_questions=questions_state["evaluation_questions"],
            evaluation_scores=scores_state["evaluation_scores"],
        )

        assert "evaluation_questions" in state
        assert "evaluation_scores" in state

    def test_answer_eval_context_has_annotations(self):
        # AnswerEvalContext inherits from VectorDBContext and LLMContext
        # Verify it has proper TypedDict annotations
        assert hasattr(AnswerEvalContext, "__annotations__")

    def test_factory_creates_independent_evaluators(self):
        factory1 = answer_eval_factory(
            score_key="score1",
            collection_key="col1",
            user_context="context1",
        )
        factory2 = answer_eval_factory(
            score_key="score2",
            collection_key="col2",
            user_context="context2",
        )
        assert factory1 is not factory2

    def test_answer_eval_result_with_integer_scores(self):
        result = AnswerEvalResult(
            feedback="Test",
            accuracy=3,  # Integer score
            completeness=4,
            relevance=5,
        )
        assert result.accuracy == 3

    def test_answer_eval_result_with_long_feedback(self):
        long_feedback = "x" * 1000
        result = AnswerEvalResult(
            feedback=long_feedback,
            accuracy=3,
            completeness=3,
            relevance=3,
        )
        assert result.feedback == long_feedback

    def test_percentage_calculations_are_floats(self):
        result = AnswerEvalResult(
            feedback="Test", accuracy=3, completeness=4, relevance=5
        )
        assert isinstance(result.accuracy_percentage, float)
        assert isinstance(result.completeness_percentage, float)
        assert isinstance(result.relevance_percentage, float)

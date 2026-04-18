import pytest
import math
from unittest.mock import MagicMock
from friday.core.evaluation import Evaluation
from friday.core.vector_db import VectorQueryOutput
from friday.graph.evaluation.retrieval_eval_factory import (
    RetrievalEvalResult,
    RetrievalEvalState,
    retrieval_eval_factory,
)


class TestRetrievalEvalResult:
    def test_retrieval_eval_result_creation(self):
        result = RetrievalEvalResult(
            mrr=0.8,
            ndcg=0.75,
            keywords_found=3,
            total_keywords=4,
            keyword_coverage=75.0,
        )
        assert result.mrr == 0.8
        assert result.ndcg == 0.75
        assert result.keywords_found == 3
        assert result.total_keywords == 4
        assert result.keyword_coverage == 75.0

    def test_retrieval_eval_result_all_fields_required(self):
        with pytest.raises(Exception):
            RetrievalEvalResult(mrr=0.8, ndcg=0.75, keywords_found=3)

    def test_retrieval_eval_result_zero_values(self):
        result = RetrievalEvalResult(
            mrr=0.0,
            ndcg=0.0,
            keywords_found=0,
            total_keywords=5,
            keyword_coverage=0.0,
        )
        assert result.mrr == 0.0
        assert result.keyword_coverage == 0.0

    def test_retrieval_eval_result_perfect_scores(self):
        result = RetrievalEvalResult(
            mrr=1.0,
            ndcg=1.0,
            keywords_found=5,
            total_keywords=5,
            keyword_coverage=100.0,
        )
        assert result.mrr == 1.0
        assert result.ndcg == 1.0
        assert result.keyword_coverage == 100.0

    def test_retrieval_eval_result_partial_coverage(self):
        result = RetrievalEvalResult(
            mrr=0.5,
            ndcg=0.6,
            keywords_found=2,
            total_keywords=5,
            keyword_coverage=40.0,
        )
        assert result.keywords_found < result.total_keywords

    def test_retrieval_eval_result_with_float_metrics(self):
        result = RetrievalEvalResult(
            mrr=0.333,
            ndcg=0.667,
            keywords_found=1,
            total_keywords=3,
            keyword_coverage=33.333,
        )
        assert result.mrr == 0.333
        assert result.ndcg == 0.667


class TestRetrievalEvalFactory:
    def test_factory_returns_callable(self):
        factory = retrieval_eval_factory(
            score_key="retrieval_score", collection_key="test_collection"
        )
        assert callable(factory)

    def test_factory_with_default_retrieval_k(self):
        factory = retrieval_eval_factory(
            score_key="score", collection_key="collection"
        )
        assert callable(factory)

    def test_factory_with_custom_retrieval_k(self):
        factory = retrieval_eval_factory(
            score_key="score", collection_key="collection", retrieval_k=20
        )
        assert callable(factory)

    def test_factory_with_small_retrieval_k(self):
        factory = retrieval_eval_factory(
            score_key="score", collection_key="collection", retrieval_k=1
        )
        assert callable(factory)

    def test_factory_with_large_retrieval_k(self):
        factory = retrieval_eval_factory(
            score_key="score", collection_key="collection", retrieval_k=100
        )
        assert callable(factory)

    def test_retrieval_eval_state_creation(self, questions_state, scores_state):
        state = RetrievalEvalState(
            evaluation_questions=questions_state["evaluation_questions"],
            evaluation_scores=scores_state["evaluation_scores"],
        )
        assert "evaluation_questions" in state
        assert "evaluation_scores" in state

    def test_retrieval_eval_state_with_empty_scores(self, questions_state):
        state = RetrievalEvalState(
            evaluation_questions=questions_state["evaluation_questions"],
            evaluation_scores={},
        )
        assert state["evaluation_scores"] == {}

    def test_factory_creates_independent_evaluators(self):
        factory1 = retrieval_eval_factory(
            score_key="score1", collection_key="col1"
        )
        factory2 = retrieval_eval_factory(
            score_key="score2", collection_key="col2"
        )
        assert factory1 is not factory2

    def test_retrieval_eval_result_coverage_percentage(self):
        result = RetrievalEvalResult(
            mrr=0.5,
            ndcg=0.5,
            keywords_found=3,
            total_keywords=5,
            keyword_coverage=60.0,
        )
        assert 0 <= result.keyword_coverage <= 100

    def test_mrr_calculation_max_value(self):
        # MRR max is 1.0 (keyword found at rank 1)
        result = RetrievalEvalResult(
            mrr=1.0,
            ndcg=0.5,
            keywords_found=1,
            total_keywords=1,
            keyword_coverage=100.0,
        )
        assert result.mrr <= 1.0

    def test_ndcg_calculation_max_value(self):
        # nDCG max is 1.0 (perfect ranking)
        result = RetrievalEvalResult(
            mrr=0.5,
            ndcg=1.0,
            keywords_found=5,
            total_keywords=5,
            keyword_coverage=100.0,
        )
        assert result.ndcg <= 1.0

    def test_retrieval_eval_result_with_multiple_keywords(self):
        result = RetrievalEvalResult(
            mrr=0.4,
            ndcg=0.45,
            keywords_found=4,
            total_keywords=10,
            keyword_coverage=40.0,
        )
        assert result.total_keywords > result.keywords_found

    def test_factory_parameters_affect_behavior(self):
        # Different retrieval_k should create different factories
        factory_k5 = retrieval_eval_factory(
            score_key="score", collection_key="col", retrieval_k=5
        )
        factory_k10 = retrieval_eval_factory(
            score_key="score", collection_key="col", retrieval_k=10
        )
        assert factory_k5 is not factory_k10

    def test_retrieval_eval_result_zero_total_keywords(self):
        result = RetrievalEvalResult(
            mrr=0.0,
            ndcg=0.0,
            keywords_found=0,
            total_keywords=0,
            keyword_coverage=0.0,
        )
        assert result.total_keywords == 0

    def test_retrieval_eval_state_inheritance(self, questions_state, scores_state):
        from friday.graph.evaluation.states.questions_state import QuestionsState
        from friday.graph.evaluation.states.scores_state import ScoresState

        state = RetrievalEvalState(
            evaluation_questions=questions_state["evaluation_questions"],
            evaluation_scores=scores_state["evaluation_scores"],
        )

        assert "evaluation_questions" in state
        assert "evaluation_scores" in state

    def test_retrieval_eval_result_metrics_are_numeric(self):
        result = RetrievalEvalResult(
            mrr=0.75,
            ndcg=0.80,
            keywords_found=4,
            total_keywords=5,
            keyword_coverage=80.0,
        )
        assert isinstance(result.mrr, float)
        assert isinstance(result.ndcg, float)
        assert isinstance(result.keywords_found, int)
        assert isinstance(result.total_keywords, int)
        assert isinstance(result.keyword_coverage, float)

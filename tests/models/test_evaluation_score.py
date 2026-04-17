import pytest
from friday.models.evaluation_score import (
    EvalQuestion,
    EvalQuestionCollection,
    EvaluationScore,
)


class TestEvalQuestion:
    def test_eval_question_initialization(self):
        question = EvalQuestion(
            question="What is AI?",
            keywords=["artificial", "intelligence"],
            answer="AI stands for artificial intelligence",
            category="general",
        )
        assert question.question == "What is AI?"
        assert question.keywords == ["artificial", "intelligence"]
        assert question.answer == "AI stands for artificial intelligence"
        assert question.category == "general"

    def test_eval_question_without_keywords(self):
        question = EvalQuestion(
            question="What is AI?",
            answer="AI stands for artificial intelligence",
            category="general",
        )
        assert question.keywords == []

    def test_eval_question_add_to_eval_question(self):
        q1 = EvalQuestion(
            question="Q1?",
            keywords=["key1"],
            answer="A1",
            category="cat1",
        )
        q2 = EvalQuestion(
            question="Q2?",
            keywords=["key2"],
            answer="A2",
            category="cat2",
        )

        result = q1 + q2
        assert isinstance(result, EvalQuestionCollection)
        assert len(result.questions) == 2
        assert result.questions[0].question == "Q1?"
        assert result.questions[1].question == "Q2?"

    def test_eval_question_add_to_collection(self):
        q1 = EvalQuestion(
            question="Q1?",
            keywords=["key1"],
            answer="A1",
            category="cat1",
        )
        collection = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                )
            ]
        )

        result = q1 + collection
        assert isinstance(result, EvalQuestionCollection)
        assert len(result.questions) == 2

    def test_eval_question_add_invalid_type_raises_exception(self):
        question = EvalQuestion(
            question="Q?",
            keywords=[],
            answer="A",
            category="cat",
        )
        with pytest.raises(Exception):
            _ = question + "invalid"


class TestEvalQuestionCollection:
    def test_collection_initialization(self):
        collection = EvalQuestionCollection()
        assert collection.questions == []

    def test_collection_initialization_with_questions(self):
        questions = [
            EvalQuestion(
                question="Q1?",
                keywords=["key1"],
                answer="A1",
                category="cat1",
            ),
            EvalQuestion(
                question="Q2?",
                keywords=["key2"],
                answer="A2",
                category="cat2",
            ),
        ]
        collection = EvalQuestionCollection(questions=questions)
        assert len(collection.questions) == 2

    def test_collection_from_questions(self):
        questions = [
            EvalQuestion(
                question="Q1?",
                keywords=["key1"],
                answer="A1",
                category="cat1",
            ),
        ]
        collection = EvalQuestionCollection.from_questions(questions)
        assert len(collection.questions) == 1
        assert collection.questions[0].question == "Q1?"

    def test_collection_add_to_collection(self):
        collection1 = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        collection2 = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                )
            ]
        )

        result = collection1 + collection2
        assert isinstance(result, EvalQuestionCollection)
        assert len(result.questions) == 2

    def test_collection_add_to_question(self):
        collection = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        question = EvalQuestion(
            question="Q2?",
            keywords=["key2"],
            answer="A2",
            category="cat2",
        )

        result = collection + question
        assert isinstance(result, EvalQuestionCollection)
        assert len(result.questions) == 2
        assert result.questions[1].question == "Q2?"

    def test_collection_add_invalid_type_raises_exception(self):
        collection = EvalQuestionCollection()
        with pytest.raises(Exception):
            _ = collection + "invalid"

    def test_collection_original_unmodified_after_add(self):
        collection1 = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        collection2 = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                )
            ]
        )

        result = collection1 + collection2
        assert len(collection1.questions) == 1
        assert len(result.questions) == 2


class TestEvaluationScore:
    def test_evaluation_score_initialization(self):
        score = EvaluationScore()
        assert isinstance(score.questions, EvalQuestionCollection)
        assert score.scores == {}

    def test_evaluation_score_with_questions(self):
        questions = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        score = EvaluationScore(questions=questions)
        assert len(score.questions.questions) == 1

    def test_evaluation_score_with_scores(self):
        scores = {"metric1": [0.9, 0.85, 0.88], "metric2": [0.95, 0.92]}
        eval_score = EvaluationScore(scores=scores)
        assert eval_score.scores == scores
        assert len(eval_score.scores["metric1"]) == 3

    def test_evaluation_score_full_initialization(self):
        questions = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        scores = {"accuracy": [0.9, 0.85]}
        eval_score = EvaluationScore(questions=questions, scores=scores)

        assert len(eval_score.questions.questions) == 1
        assert eval_score.scores == scores

    def test_evaluation_score_with_multiple_metrics(self):
        scores = {
            "precision": [0.95, 0.93, 0.91],
            "recall": [0.88, 0.90, 0.92],
            "f1": [0.91, 0.91, 0.91],
        }
        eval_score = EvaluationScore(scores=scores)
        assert len(eval_score.scores) == 3
        assert eval_score.scores["precision"] == [0.95, 0.93, 0.91]

    def test_evaluation_score_with_complex_score_values(self):
        scores = {
            "metrics": ["score1", "score2", 0.95],
            "mixed": [1, "string", 3.14, True],
        }
        eval_score = EvaluationScore(scores=scores)
        assert eval_score.scores["metrics"][0] == "score1"
        assert eval_score.scores["mixed"][1] == "string"

    def test_evaluation_score_serialization(self):
        questions = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        scores = {"accuracy": [0.9]}
        eval_score = EvaluationScore(questions=questions, scores=scores)

        data = eval_score.model_dump()
        assert len(data["questions"]["questions"]) == 1
        assert data["scores"] == scores

    def test_evaluation_score_deserialization(self):
        data = {
            "questions": {
                "questions": [
                    {
                        "question": "Q1?",
                        "keywords": ["key1"],
                        "answer": "A1",
                        "category": "cat1",
                    }
                ]
            },
            "scores": {"accuracy": [0.9]},
        }
        eval_score = EvaluationScore(**data)
        assert len(eval_score.questions.questions) == 1
        assert eval_score.scores["accuracy"] == [0.9]

    def test_evaluation_score_empty_scores(self):
        questions = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=[],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        eval_score = EvaluationScore(questions=questions)
        assert eval_score.scores == {}

    def test_evaluation_score_multiple_questions_and_scores(self):
        questions = EvalQuestionCollection(
            questions=[
                EvalQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                ),
                EvalQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                ),
            ]
        )
        scores = {
            "accuracy": [0.9, 0.85],
            "precision": [0.95, 0.92],
        }
        eval_score = EvaluationScore(questions=questions, scores=scores)

        assert len(eval_score.questions.questions) == 2
        assert len(eval_score.scores) == 2

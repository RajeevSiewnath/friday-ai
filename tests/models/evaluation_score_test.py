from models.evaluation_score import (
    EvalQuestion,
    EvalQuestionCollection,
    EvaluationScore,
)


def test_eval_question_creation():
    """Test that EvalQuestion can be created with all fields."""
    question = EvalQuestion(
        question="What is Python?",
        keywords=["programming", "language"],
        answer="Python is a programming language",
        category="basics",
    )

    assert question.question == "What is Python?"
    assert question.keywords == ["programming", "language"]
    assert question.answer == "Python is a programming language"
    assert question.category == "basics"


def test_eval_question_default_keywords():
    """Test that EvalQuestion defaults to empty keywords."""
    question = EvalQuestion(question="Test?", answer="Answer", category="test")

    assert question.keywords == []


def test_eval_question_add_with_another_question():
    """Test that EvalQuestion + EvalQuestion returns EvalQuestionCollection."""
    q1 = EvalQuestion(question="Q1?", answer="A1", category="cat1")
    q2 = EvalQuestion(question="Q2?", answer="A2", category="cat2")

    result = q1 + q2

    assert isinstance(result, EvalQuestionCollection)
    assert len(result.questions) == 2
    assert result.questions[0].question == "Q1?"


def test_eval_question_add_with_collection():
    """Test that EvalQuestion + EvalQuestionCollection returns EvalQuestionCollection."""
    q = EvalQuestion(question="Q1?", answer="A1", category="cat1")
    collection = EvalQuestionCollection(
        questions=[
            EvalQuestion(question="Q2?", answer="A2", category="cat2"),
            EvalQuestion(question="Q3?", answer="A3", category="cat3"),
        ]
    )

    result = q + collection

    assert isinstance(result, EvalQuestionCollection)
    assert len(result.questions) == 3
    assert result.questions[0].question == "Q1?"


def test_eval_question_add_with_invalid_type():
    """Test that EvalQuestion + invalid type raises exception."""
    q = EvalQuestion(question="Test?", answer="Answer", category="test")

    try:
        _ = q + "invalid"
        assert False, "Should have raised exception"
    except Exception as e:
        assert "cannot add" in str(e)


def test_eval_question_collection_creation():
    """Test that EvalQuestionCollection can be created with questions."""
    questions = [
        EvalQuestion(question="Q1?", answer="A1", category="cat1"),
        EvalQuestion(question="Q2?", answer="A2", category="cat2"),
    ]

    collection = EvalQuestionCollection(questions=questions)

    assert len(collection.questions) == 2
    assert collection.questions[0].question == "Q1?"


def test_eval_question_collection_default_empty():
    """Test that EvalQuestionCollection defaults to empty list."""
    collection = EvalQuestionCollection()
    assert collection.questions == []


def test_eval_question_collection_from_questions():
    """Test EvalQuestionCollection.from_questions() class method."""
    questions = [
        EvalQuestion(question="Q1?", answer="A1", category="cat1"),
        EvalQuestion(question="Q2?", answer="A2", category="cat2"),
    ]

    collection = EvalQuestionCollection.from_questions(questions)

    assert len(collection.questions) == 2
    assert collection.questions[0].question == "Q1?"


def test_eval_question_collection_add_with_another_collection():
    """Test that EvalQuestionCollection + EvalQuestionCollection combines questions."""
    col1 = EvalQuestionCollection(
        questions=[EvalQuestion(question="Q1?", answer="A1", category="cat1")]
    )
    col2 = EvalQuestionCollection(
        questions=[EvalQuestion(question="Q2?", answer="A2", category="cat2")]
    )

    result = col1 + col2

    assert isinstance(result, EvalQuestionCollection)
    assert len(result.questions) == 2


def test_eval_question_collection_add_with_question():
    """Test that EvalQuestionCollection + EvalQuestion appends the question."""
    collection = EvalQuestionCollection(
        questions=[EvalQuestion(question="Q1?", answer="A1", category="cat1")]
    )
    q = EvalQuestion(question="Q2?", answer="A2", category="cat2")

    result = collection + q

    assert isinstance(result, EvalQuestionCollection)
    assert len(result.questions) == 2
    assert result.questions[-1].question == "Q2?"


def test_eval_question_collection_add_with_invalid_type():
    """Test that EvalQuestionCollection + invalid type raises exception."""
    collection = EvalQuestionCollection()

    try:
        _ = collection + "invalid"
        assert False, "Should have raised exception"
    except Exception as e:
        assert "cannot add" in str(e)


def test_evaluation_score_creation():
    """Test that EvaluationScore can be created."""
    eval_score = EvaluationScore()

    assert isinstance(eval_score.questions, EvalQuestionCollection)
    assert eval_score.questions.questions == []
    assert eval_score.scores == []


def test_evaluation_score_with_questions():
    """Test that EvaluationScore can be initialized with questions."""
    questions = EvalQuestionCollection(
        questions=[
            EvalQuestion(question="Q1?", answer="A1", category="cat1"),
            EvalQuestion(question="Q2?", answer="A2", category="cat2"),
        ]
    )

    eval_score = EvaluationScore(questions=questions)

    assert len(eval_score.questions.questions) == 2


def test_evaluation_score_with_scores():
    """Test that EvaluationScore can be initialized with scores."""
    scores = [0.9, 0.85, 0.95]

    eval_score = EvaluationScore(scores=scores)

    assert eval_score.scores == [0.9, 0.85, 0.95]


def test_evaluation_score_with_questions_and_scores():
    """Test that EvaluationScore can be initialized with both questions and scores."""
    questions = EvalQuestionCollection(
        questions=[
            EvalQuestion(question="Q1?", answer="A1", category="cat1"),
        ]
    )
    scores = [0.9]

    eval_score = EvaluationScore(questions=questions, scores=scores)

    assert len(eval_score.questions.questions) == 1
    assert eval_score.scores == [0.9]


def test_evaluation_score_complex_data():
    """Test EvaluationScore with complex score data."""
    questions = EvalQuestionCollection(
        questions=[
            EvalQuestion(
                question="What is Python?",
                keywords=["language"],
                answer="A programming language",
                category="basics",
            ),
        ]
    )

    scores = [
        {"retrieval_score": 0.9, "relevance_score": 0.85},
        {"retrieval_score": 0.8, "relevance_score": 0.88},
    ]

    eval_score = EvaluationScore(questions=questions, scores=scores)

    assert len(eval_score.scores) == 2
    assert eval_score.scores[0]["retrieval_score"] == 0.9

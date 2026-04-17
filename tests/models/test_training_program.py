import pytest
from friday.models.training_program import (
    TrainingQuestion,
    TrainingQuestionCollection,
    TrainingProgram,
)


class TestTrainingQuestion:
    def test_training_question_initialization(self):
        question = TrainingQuestion(
            question="What is ML?",
            keywords=["machine", "learning"],
            answer="ML is machine learning",
            category="general",
        )
        assert question.question == "What is ML?"
        assert question.keywords == ["machine", "learning"]
        assert question.answer == "ML is machine learning"
        assert question.category == "general"

    def test_training_question_without_keywords(self):
        question = TrainingQuestion(
            question="What is ML?",
            answer="ML is machine learning",
            category="general",
        )
        assert question.keywords == []

    def test_training_question_add_to_training_question(self):
        q1 = TrainingQuestion(
            question="Q1?",
            keywords=["key1"],
            answer="A1",
            category="cat1",
        )
        q2 = TrainingQuestion(
            question="Q2?",
            keywords=["key2"],
            answer="A2",
            category="cat2",
        )

        result = q1 + q2
        assert isinstance(result, TrainingQuestionCollection)
        assert len(result.questions) == 2

    def test_training_question_add_to_collection(self):
        q1 = TrainingQuestion(
            question="Q1?",
            keywords=["key1"],
            answer="A1",
            category="cat1",
        )
        collection = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                )
            ]
        )

        result = q1 + collection
        assert isinstance(result, TrainingQuestionCollection)
        assert len(result.questions) == 2

    def test_training_question_add_invalid_type_raises_exception(self):
        question = TrainingQuestion(
            question="Q?",
            keywords=[],
            answer="A",
            category="cat",
        )
        with pytest.raises(Exception):
            _ = question + "invalid"


class TestTrainingQuestionCollection:
    def test_collection_initialization(self):
        collection = TrainingQuestionCollection()
        assert collection.questions == []

    def test_collection_initialization_with_questions(self):
        questions = [
            TrainingQuestion(
                question="Q1?",
                keywords=["key1"],
                answer="A1",
                category="cat1",
            ),
            TrainingQuestion(
                question="Q2?",
                keywords=["key2"],
                answer="A2",
                category="cat2",
            ),
        ]
        collection = TrainingQuestionCollection(questions=questions)
        assert len(collection.questions) == 2

    def test_collection_from_questions(self):
        questions = [
            TrainingQuestion(
                question="Q1?",
                keywords=["key1"],
                answer="A1",
                category="cat1",
            ),
        ]
        collection = TrainingQuestionCollection.from_questions(questions)
        assert len(collection.questions) == 1

    def test_collection_add_to_collection(self):
        collection1 = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        collection2 = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                )
            ]
        )

        result = collection1 + collection2
        assert isinstance(result, TrainingQuestionCollection)
        assert len(result.questions) == 2

    def test_collection_add_to_question(self):
        collection = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        question = TrainingQuestion(
            question="Q2?",
            keywords=["key2"],
            answer="A2",
            category="cat2",
        )

        result = collection + question
        assert isinstance(result, TrainingQuestionCollection)
        assert len(result.questions) == 2

    def test_collection_add_invalid_type_raises_exception(self):
        collection = TrainingQuestionCollection()
        with pytest.raises(Exception):
            _ = collection + "invalid"

    def test_collection_original_unmodified_after_add(self):
        collection1 = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        collection2 = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
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


class TestTrainingProgram:
    def test_training_program_initialization(self):
        program = TrainingProgram()
        assert isinstance(program.full, TrainingQuestionCollection)
        assert isinstance(program.training, TrainingQuestionCollection)
        assert isinstance(program.validating, TrainingQuestionCollection)
        assert isinstance(program.testing, TrainingQuestionCollection)
        assert program.statistics == {}

    def test_training_program_with_all_splits(self):
        full = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        training = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                )
            ]
        )
        validating = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q3?",
                    keywords=["key3"],
                    answer="A3",
                    category="cat3",
                )
            ]
        )
        testing = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q4?",
                    keywords=["key4"],
                    answer="A4",
                    category="cat4",
                )
            ]
        )

        program = TrainingProgram(
            full=full,
            training=training,
            validating=validating,
            testing=testing,
        )

        assert len(program.full.questions) == 1
        assert len(program.training.questions) == 1
        assert len(program.validating.questions) == 1
        assert len(program.testing.questions) == 1

    def test_training_program_with_statistics(self):
        stats = {
            "train_accuracy": 0.95,
            "val_accuracy": 0.92,
            "test_accuracy": 0.90,
            "training_time": 3600,
        }
        program = TrainingProgram(statistics=stats)
        assert program.statistics == stats
        assert program.statistics["train_accuracy"] == 0.95

    def test_training_program_full_initialization(self):
        full = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                ),
                TrainingQuestion(
                    question="Q2?",
                    keywords=["key2"],
                    answer="A2",
                    category="cat2",
                ),
                TrainingQuestion(
                    question="Q3?",
                    keywords=["key3"],
                    answer="A3",
                    category="cat3",
                ),
                TrainingQuestion(
                    question="Q4?",
                    keywords=["key4"],
                    answer="A4",
                    category="cat4",
                ),
            ]
        )
        training = TrainingQuestionCollection(
            questions=full.questions[:3]
        )
        validating = TrainingQuestionCollection(
            questions=full.questions[3:4]
        )

        program = TrainingProgram(
            full=full,
            training=training,
            validating=validating,
        )

        assert len(program.full.questions) == 4
        assert len(program.training.questions) == 3
        assert len(program.validating.questions) == 1

    def test_training_program_with_complex_statistics(self):
        stats = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.93,
            "f1_score": 0.925,
            "training_steps": 1000,
            "training_time_seconds": 3600.5,
            "validation_loss": 0.125,
        }
        program = TrainingProgram(statistics=stats)
        assert len(program.statistics) == 7
        assert program.statistics["f1_score"] == 0.925

    def test_training_program_empty_statistics(self):
        program = TrainingProgram()
        assert program.statistics == {}

    def test_training_program_serialization(self):
        full = TrainingQuestionCollection(
            questions=[
                TrainingQuestion(
                    question="Q1?",
                    keywords=["key1"],
                    answer="A1",
                    category="cat1",
                )
            ]
        )
        stats = {"accuracy": 0.95}
        program = TrainingProgram(full=full, statistics=stats)

        data = program.model_dump()
        assert len(data["full"]["questions"]) == 1
        assert data["statistics"] == stats

    def test_training_program_deserialization(self):
        data = {
            "full": {
                "questions": [
                    {
                        "question": "Q1?",
                        "keywords": ["key1"],
                        "answer": "A1",
                        "category": "cat1",
                    }
                ]
            },
            "training": {"questions": []},
            "validating": {"questions": []},
            "testing": {"questions": []},
            "statistics": {"accuracy": 0.95},
        }
        program = TrainingProgram(**data)
        assert len(program.full.questions) == 1
        assert program.statistics["accuracy"] == 0.95

    def test_training_program_access_split_questions(self):
        q1 = TrainingQuestion(
            question="Q1?",
            keywords=["key1"],
            answer="A1",
            category="cat1",
        )
        q2 = TrainingQuestion(
            question="Q2?",
            keywords=["key2"],
            answer="A2",
            category="cat2",
        )

        training = TrainingQuestionCollection(questions=[q1])
        testing = TrainingQuestionCollection(questions=[q2])

        program = TrainingProgram(training=training, testing=testing)

        assert program.training.questions[0].question == "Q1?"
        assert program.testing.questions[0].question == "Q2?"

    def test_training_program_with_questions_in_each_split(self):
        q1 = TrainingQuestion(
            question="Q1?",
            keywords=["key1"],
            answer="A1",
            category="cat1",
        )
        q2 = TrainingQuestion(
            question="Q2?",
            keywords=["key2"],
            answer="A2",
            category="cat2",
        )
        q3 = TrainingQuestion(
            question="Q3?",
            keywords=["key3"],
            answer="A3",
            category="cat3",
        )

        program = TrainingProgram(
            full=TrainingQuestionCollection(questions=[q1, q2, q3]),
            training=TrainingQuestionCollection(questions=[q1, q2]),
            testing=TrainingQuestionCollection(questions=[q3]),
        )

        assert len(program.full.questions) == 3
        assert len(program.training.questions) == 2
        assert len(program.testing.questions) == 1

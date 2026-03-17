import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from evaluation_pipes.questions_loader import QuestionsLoader
from models.evaluation_score import EvaluationScore


@pytest.fixture
def evaluation_loader_pipe():
    return QuestionsLoader("rag_evaluation.json", 2)


@pytest.fixture
def evaluation_loader_pipe_full():
    return QuestionsLoader("rag_evaluation.json")


@pytest.fixture
def evaluation_pipe_arg():
    return EvaluationScore()

import pytest
from core.llm import LLM
from core.prompt_context import PromptContext
from core.vector_db import VectorDB
from evaluation_pipes.questions_loader import QuestionsLoader
from pipelines.evaluation_pipeline import EvaluationPipeline, EvaluationScore


@pytest.fixture
def evaluation_loader_pipe():
    return QuestionsLoader("rag_evaluation.json", 2)


@pytest.fixture
def evaluation_loader_pipe_full():
    return QuestionsLoader("rag_evaluation.json")


@pytest.fixture
def evaluation_pipe_arg():
    return EvaluationScore()


@pytest.fixture(scope="function")
def evaluation_pipeline(llm: LLM, prompt_context: PromptContext, vector_db: VectorDB):
    return EvaluationPipeline(
        llm=llm,
        prompt_context=prompt_context,
        vector_db=vector_db,
    )

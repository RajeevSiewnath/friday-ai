from evaluation_pipes.questions_loader import QuestionsLoader
from evaluation_pipes.retrieval_eval import RetrievalEval
from pipelines.evaluation_pipeline import (
    EvaluationPipeline,
    EvaluationScore,
)


def test_answer_eval(
    evaluation_pipeline: EvaluationPipeline,
    evaluation_loader_pipe: QuestionsLoader,
    evaluation_pipe_arg: EvaluationScore,
):
    evaluation_score: EvaluationScore = evaluation_pipeline.add(
        evaluation_loader_pipe,
        RetrievalEval("test"),
    ).run(evaluation_pipe_arg)
    assert len(evaluation_score.questions.questions) > 0
    assert len(evaluation_score.scores["test"]) > 0

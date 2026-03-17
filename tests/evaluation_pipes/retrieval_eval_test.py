from evaluation_pipes.questions_loader import QuestionsLoader
from evaluation_pipes.retrieval_eval import RetrievalEval
from models.evaluation_score import EvaluationScore
from pipelines.pipeline_factory import PipelineFactory


def test_answer_eval(
    pipeline_factory: PipelineFactory,
    evaluation_loader_pipe: QuestionsLoader,
    evaluation_pipe_arg: EvaluationScore,
):
    evaluation_score = pipeline_factory.make(
        evaluation_loader_pipe,
        RetrievalEval("test"),
    ).run(evaluation_pipe_arg)
    assert len(evaluation_score.questions.questions) > 0
    assert len(evaluation_score.scores["test"]) > 0

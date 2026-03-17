from evaluation_pipes.questions_loader import QuestionsLoader
from models.evaluation_score import EvaluationScore
from pipelines.pipeline_factory import PipelineFactory


def test_questions_loader(
    pipeline_factory: PipelineFactory,
    evaluation_loader_pipe_full: QuestionsLoader,
    evaluation_pipe_arg: EvaluationScore,
):
    evaluation_score = pipeline_factory.make(
        evaluation_loader_pipe_full,
    ).run(evaluation_pipe_arg)
    assert len(evaluation_score.questions.questions) > 0

from evaluation_pipes.questions_loader import QuestionsLoader
from pipelines.evaluation_pipeline import EvaluationPipeline, EvaluationScore


def test_questions_loader(
    evaluation_pipeline: EvaluationPipeline,
    evaluation_loader_pipe_full: QuestionsLoader,
    evaluation_pipe_arg: EvaluationScore,
):
    evaluation_score: EvaluationScore = evaluation_pipeline.add(
        evaluation_loader_pipe_full,
    ).run(evaluation_pipe_arg)
    assert len(evaluation_score.questions.questions) > 0

from evaluation_pipes.answer_eval import AnswerEval
from evaluation_pipes.questions_loader import QuestionsLoader
from pipelines.abstract_pipeline import PipeArg
from pipelines.evaluation_pipeline import (
    EvaluationPipeline,
    EvaluationScore,
)


def test_answer_eval(
    evaluation_pipeline: EvaluationPipeline,
    evaluation_loader_pipe: QuestionsLoader,
    evaluation_pipe_arg: PipeArg[EvaluationScore],
):
    evaluation_score: EvaluationScore = evaluation_pipeline.add(
        evaluation_loader_pipe,
        AnswerEval(),
    ).run(evaluation_pipe_arg)
    assert len(evaluation_score.questions.questions) > 0
    assert len(evaluation_score.scores) > 0

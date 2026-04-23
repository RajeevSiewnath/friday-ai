from dataclasses import dataclass
from logging import LoggerAdapter
from tqdm import tqdm
from langgraph.runtime import Runtime
from friday.core.evaluation import Evaluation
from friday.graph.evaluation.states.questions_state import QuestionsState
from friday.graph.evaluation.states.scores_state import ScoresState
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.contexts.vector_db_context import VectorDBContext
from pydantic import BaseModel, Field

from friday.loggers.logger import Logger


class AnswerEvalResult(BaseModel):
    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )

    @property
    def accuracy_percentage(self):
        return (self.accuracy - 1) / 4

    @property
    def completeness_percentage(self):
        return (self.completeness - 1) / 4

    @property
    def relevance_percentage(self):
        return (self.relevance - 1) / 4


class AnswerEvalState(QuestionsState, ScoresState):
    pass


@dataclass
class AnswerEvalContext(VectorDBContext, LLMContext):
    pass


def answer_eval_factory(
    score_key: str, collection_key: str, user_context: str, retrieval_k: int = 10
):
    async def answer_question(
        question: str, context: AnswerEvalContext, logger: LoggerAdapter
    ):
        system_prompt = """
{user_context}
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""
        results = context.vector_db[collection_key].query(question, retrieval_k)

        context = "\n\n".join(result.document for result in results)
        input = [
            {
                "role": "system",
                "content": system_prompt.format(
                    context=context, user_context=user_context
                ),
            }
        ] + [{"role": "user", "content": question}]
        logger.debug("input (answer): %s", lambda: input)

        response = await context.llm.invoke(input)
        logger.debug("response (answer): %s", lambda: response)

        return response.output_text

    async def evaluate_answer(
        test: Evaluation, context: AnswerEvalContext, logger: LoggerAdapter
    ) -> AnswerEvalResult:
        generated_answer = await answer_question(test.question, context, logger)

        judge_messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
            },
            {
                "role": "user",
                "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
            },
        ]
        logger.debug("input (judge): %s", lambda: judge_messages)

        judge_response = await context.llm.invoke(judge_messages, AnswerEvalResult)
        logger.debug("response (judge): %s", lambda: judge_response)

        return judge_response.output_parsed

    async def answer_eval(state: AnswerEvalState, runtime: Runtime[AnswerEvalContext]):
        logger = Logger.get_logger("node.answer_eval")
        logger.info("evaluating answers to questions")
        logger.debug("questions: %s", lambda: state["questions"])

        responses = [
            await evaluate_answer(question, runtime.context, logger)
            for question in tqdm(state["questions"])
        ]
        return {"evaluation_scores": {score_key: responses}}

    return answer_eval

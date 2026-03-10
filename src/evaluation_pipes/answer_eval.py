from tqdm import tqdm
from pipelines.abstract_pipeline import AbstractPipe, PipeArg
from pydantic import BaseModel, Field
from models.evaluation_score import EvalQuestion, EvaluationScore


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


class AnswerEval(AbstractPipe[EvaluationScore]):

    def __init__(
        self,
        retrieval_k: int = 10,
    ):
        super().__init__()
        self.system_prompt = """
  {user_context}
  If relevant, use the given context to answer any question.
  If you don't know the answer, say so.
  Context:
  {context}
  """
        self.retrieval_k = retrieval_k

    def combined_question(self, question: str, arg: PipeArg[EvaluationScore]):
        prior = "\n".join(
            m.get("content")
            for m in arg.prompt_context.history
            if m.get("role") == "user"
        )
        return prior + "\n" + question

    def answer_question(self, question: str, arg: PipeArg[EvaluationScore]):
        combined = self.combined_question(question, arg)
        results = arg.vector_db.query(combined, self.retrieval_k)

        context = "\n\n".join(doc.content for doc in results.documents)
        input = (
            [
                {
                    "role": "system",
                    "content": self.system_prompt.format(
                        context=context, user_context=arg.prompt_context.user_context
                    ),
                }
            ]
            + arg.prompt_context.history
            + [{"role": "user", "content": question}]
        )
        response = arg.llm.invoke(input)
        return response

    def evaluate_answer(self, test: EvalQuestion, arg: PipeArg[EvaluationScore]):
        generated_answer = self.answer_question(test.question, arg)

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

        judge_response = arg.llm.invoke(judge_messages, AnswerEvalResult)

        return judge_response

    def pipe(self, arg):
        for question in tqdm(arg.input.questions.questions):
            judge_response = self.evaluate_answer(question, arg)
            arg.input.scores.append(judge_response)
        return arg.input

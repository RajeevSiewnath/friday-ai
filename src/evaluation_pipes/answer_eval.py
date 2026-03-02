from chromadb import Collection
import chromadb
from tqdm import tqdm
from core.llm import embedding, invoke
from evaluation.test_loader import TestQuestion
from evaluation_pipes.questions_loader import QuestionsLoader
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.evaluation_pipeline import EvaluationScore
from pydantic import BaseModel, Field
from chromadb.config import Settings


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

    system_prompt = """
  You are a personal job agent for Rajeev Siewnath. 
  You provide information about his curriculum vitae to the user.
  If relevant, use the given context to answer any question.
  If you don't know the answer, say so.
  Context:
  {context}
  """
    collection: Collection
    retrieval_k: int

    def __init__(self, collection: Collection, retrieval_k: int = 10):
        super().__init__()
        self.collection = collection
        self.retrieval_k = retrieval_k

    def combined_question(self, question: str, history: list[dict] = []):
        prior = "\n".join(m["content"] for m in history if m["role"] == "user")
        return prior + "\n" + question

    def answer_question(self, question: str, history: list[dict] = []):
        combined = self.combined_question(question, history)
        query = embedding(combined)
        results = self.collection.query(
            query_embeddings=query, n_results=self.retrieval_k
        )

        context = "\n\n".join(doc for doc in results["documents"][0])
        input = (
            [{"role": "system", "content": self.system_prompt.format(context=context)}]
            + history
            + [{"role": "user", "content": question}]
        )
        response = invoke(input)
        return response

    def evaluate_answer(self, test: TestQuestion):
        generated_answer = self.answer_question(test.question)

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

        judge_response: AnswerEvalResult = invoke(
            input=judge_messages, response_format=AnswerEvalResult
        )

        return judge_response

    def pipe(self, input):
        for question in tqdm(input.questions.questions):
            judge_response = self.evaluate_answer(question)
            input.scores.append(judge_response)
        return input


if __name__ == "__main__":
    chroma = chromadb.Client(Settings(is_persistent=True))
    collection: Collection = chroma.get_collection(name="cv-rajeev-siewnath")
    eval_score = EvaluationScore()
    eval_score = QuestionsLoader("rag_evaluation.json").pipe(eval_score)
    print(AnswerEval(collection).pipe(eval_score))

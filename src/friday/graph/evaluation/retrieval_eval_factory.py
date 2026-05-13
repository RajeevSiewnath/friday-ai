import math
from pydantic import BaseModel, Field
from tqdm import tqdm
from langgraph.runtime import Runtime
from friday.core.evaluation import Evaluation
from friday.core.vector_db import VectorQueryOutput
from friday.graph.evaluation.states.questions_state import QuestionsState
from friday.graph.evaluation.states.scores_state import ScoresState
from friday.graph.query.contexts.vector_db_context import VectorDBContext
from friday.loggers.logger import Logger


class RetrievalEvalResult(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance)"
    )
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class RetrievalEvalState(QuestionsState, ScoresState):
    pass


def retrieval_eval_factory(score_key: str, collection_key: str, retrieval_k: int = 10):

    def calculate_mrr(keyword: str, vector_query_outputs: list[VectorQueryOutput]):
        """Calculate reciprocal rank for a single keyword (case-insensitive)."""
        keyword_lower = keyword.lower()
        documents = [v.document for v in vector_query_outputs]
        for rank, doc in enumerate(documents, start=1):
            if keyword_lower in doc.lower():
                return 1.0 / rank
        return 0.0

    def calculate_dcg(relevances: list[int], k: int):
        """Calculate Discounted Cumulative Gain."""
        dcg = 0.0
        for i in range(min(k, len(relevances))):
            dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
        return dcg

    def calculate_ndcg(keyword: str, vector_query_outputs: list[VectorQueryOutput]):
        """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
        keyword_lower = keyword.lower()

        # Binary relevance: 1 if keyword found, 0 otherwise
        relevances = [
            1 if keyword_lower in v.document.lower() else 0
            for v in vector_query_outputs[:retrieval_k]
        ]

        # DCG
        dcg = calculate_dcg(relevances, retrieval_k)

        # Ideal DCG (best case: keyword in first position)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = calculate_dcg(ideal_relevances, retrieval_k)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_retrieval(test: Evaluation, context: VectorDBContext):
        """
        Evaluate retrieval performance for a test question.

        Args:
            test: TestQuestion object containing question and keywords
            k: Number of top documents to retrieve (default 10)

        Returns:
            RetrievalEval object with MRR, nDCG, and keyword coverage metrics
        """
        # Retrieve documents using shared answer module
        retrieved_docs = context.vector_db[collection_key].query(test.question)

        # Calculate MRR (average across all keywords)
        mrr_scores = [
            calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords
        ]
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

        # Calculate nDCG (average across all keywords)
        ndcg_scores = [
            calculate_ndcg(keyword, retrieved_docs) for keyword in test.keywords
        ]
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

        # Calculate keyword coverage
        keywords_found = sum(1 for score in mrr_scores if score > 0)
        total_keywords = len(test.keywords)
        keyword_coverage = (
            (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0
        )

        return RetrievalEvalResult(
            mrr=avg_mrr,
            ndcg=avg_ndcg,
            keywords_found=keywords_found,
            total_keywords=total_keywords,
            keyword_coverage=keyword_coverage,
        )

    def retrieval_eval(state: RetrievalEvalState, runtime: Runtime[VectorDBContext]):
        logger = Logger.get_logger("node.retrieval_eval")
        logger.debug("evaluating retrieval performance for questions")
        logger.trace("questions: %s", lambda: state["questions"])

        responses = [
            evaluate_retrieval(question, runtime.context)
            for question in tqdm(state["questions"])
        ]

        logger.trace("retrieval evaluation results: %s", lambda: responses)
        return {"evaluation_scores": {score_key: responses}}

    return retrieval_eval

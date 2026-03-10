import math
from pydantic import BaseModel, Field
from tqdm import tqdm
from pipelines.abstract_pipeline import AbstractPipe, PipeArg
from models.document import DocumentCollection
from models.evaluation_score import EvalQuestion, EvaluationScore


class RetrievalEvalResult(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance)"
    )
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class RetrievalEval(AbstractPipe[EvaluationScore]):

    def __init__(self, retrieval_k: int = 10):
        super().__init__()
        self.retrieval_k = retrieval_k

    def calculate_mrr(self, keyword: str, retrieved_docs: DocumentCollection):
        """Calculate reciprocal rank for a single keyword (case-insensitive)."""
        keyword_lower = keyword.lower()
        for rank, doc in enumerate(retrieved_docs.documents, start=1):
            if keyword_lower in doc.content.lower():
                return 1.0 / rank
        return 0.0

    def calculate_dcg(self, relevances: list[int], k: int):
        """Calculate Discounted Cumulative Gain."""
        dcg = 0.0
        for i in range(min(k, len(relevances))):
            dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
        return dcg

    def calculate_ndcg(self, keyword: str, retrieved_docs: DocumentCollection):
        """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
        keyword_lower = keyword.lower()

        # Binary relevance: 1 if keyword found, 0 otherwise
        relevances = [
            1 if keyword_lower in doc.content.lower() else 0
            for doc in retrieved_docs.documents[:10]
        ]

        # DCG
        dcg = self.calculate_dcg(relevances, 10)

        # Ideal DCG (best case: keyword in first position)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.calculate_dcg(ideal_relevances, 10)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_retrieval(self, test: EvalQuestion, arg: PipeArg[EvaluationScore]):
        """
        Evaluate retrieval performance for a test question.

        Args:
            test: TestQuestion object containing question and keywords
            k: Number of top documents to retrieve (default 10)

        Returns:
            RetrievalEval object with MRR, nDCG, and keyword coverage metrics
        """
        # Retrieve documents using shared answer module
        retrieved_docs = arg.vector_db.query(test.question)

        # Calculate MRR (average across all keywords)
        mrr_scores = [
            self.calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords
        ]
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

        # Calculate nDCG (average across all keywords)
        ndcg_scores = [
            self.calculate_ndcg(keyword, retrieved_docs) for keyword in test.keywords
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

    def pipe(self, arg):
        for question in tqdm(arg.input.questions.questions):
            judge_response = self.evaluate_retrieval(question, arg)
            arg.input.scores.append(judge_response)
        return arg.input

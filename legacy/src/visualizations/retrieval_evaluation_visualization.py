from evaluation_pipes.retrieval_eval import RetrievalEvalResult
from models.evaluation_score import EvaluationScore
from visualizations.abstract_visualization import AbstractVisualization
import plotly.graph_objects as go
import random


class RetrievalEvaluationVisualization(AbstractVisualization):

    def __init__(self, title: str, evaluation_score: EvaluationScore, keys: list[str]):
        super().__init__(title)
        self.evaluation_score = evaluation_score
        self.keys = keys

    def draw(self):
        sets = []
        mrr = []
        ndcg = []
        keyword_coverage = []
        for key in self.keys:
            sets.append(key)
            data: list[RetrievalEvalResult] = self.evaluation_score.scores.get(key, [])
            mrr.append(sum(score.mrr for score in data) / len(data))
            ndcg.append(sum(score.ndcg for score in data) / len(data))
            keyword_coverage.append(
                sum(score.keyword_coverage / 100 for score in data) / len(data)
            )

        self.fig.add_trace(
            go.Bar(
                name="MRR",
                x=sets,
                y=mrr,
                marker_color=f"rgb({random.randint(50,220)}, {random.randint(50,220)}, {random.randint(50,220)})",
            )
        )
        self.fig.add_trace(
            go.Bar(
                name="nDCG",
                x=sets,
                y=ndcg,
                marker_color=f"rgb({random.randint(50,220)}, {random.randint(50,220)}, {random.randint(50,220)})",
            )
        )
        self.fig.add_trace(
            go.Bar(
                name="kw cov.",
                x=sets,
                y=keyword_coverage,
                marker_color=f"rgb({random.randint(50,220)}, {random.randint(50,220)}, {random.randint(50,220)})",
            )
        )

        self.fig.update_layout(
            title=self.title,
            barmode="group",
            yaxis=dict(range=[0, 1], title="Accuracy"),
            xaxis=dict(title="Test set"),
            legend_title="Metrics",
            template="plotly_white",
        )

from friday.graph.evaluation.answer_eval_factory import AnswerEvalResult
from friday.models.evaluation_score import EvaluationScore
from friday.visualizations.abstract_visualization import AbstractVisualization
import plotly.graph_objects as go
import random


class AnswerEvaluationVisualization(AbstractVisualization):

    def __init__(self, title: str, evaluation_score: EvaluationScore, keys: list[str]):
        super().__init__(title)
        self.evaluation_score = evaluation_score
        self.keys = keys

    def draw(self):
        sets = []
        acc = []
        rel = []
        comp = []
        for key in self.keys:
            sets.append(key)
            data: list[AnswerEvalResult] = self.evaluation_score.scores.get(key, [])
            acc.append(sum(score.accuracy_percentage for score in data) / len(data))
            rel.append(sum(score.relevance_percentage for score in data) / len(data))
            comp.append(
                sum(score.completeness_percentage for score in data) / len(data)
            )

        self.fig.add_trace(
            go.Bar(
                name="Accuracy",
                x=sets,
                y=acc,
                marker_color=f"rgb({random.randint(50,220)}, {random.randint(50,220)}, {random.randint(50,220)})",
            )
        )
        self.fig.add_trace(
            go.Bar(
                name="Relevance",
                x=sets,
                y=rel,
                marker_color=f"rgb({random.randint(50,220)}, {random.randint(50,220)}, {random.randint(50,220)})",
            )
        )
        self.fig.add_trace(
            go.Bar(
                name="Completeness",
                x=sets,
                y=comp,
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

from chromadb import Collection
from evaluation_pipes.answer_eval import AnswerEvalResult
from models.evaluation_score import EvaluationScore
from visualizations.abstract_visualization import AbstractVisualization
import plotly.graph_objects as go
import random
from sklearn.manifold import TSNE
import numpy as np


class VectordbTnseVisualization(AbstractVisualization):

    def __init__(self, title: str, collection: Collection, is_3d=False):
        super().__init__(title)
        self.collection = collection
        self._highlight_ids: list[str] = []
        self.is_3d = is_3d

    @property
    def highlight_ids(self):
        return self._highlight_ids

    @highlight_ids.setter
    def highlight_ids(self, value: list[str]):
        collection_data = self.collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        self._highlight_ids = value
        self.fig.data[0].marker.size = [
            8 if id in value else 5 for id in collection_data["ids"]
        ]

    def draw(self):
        collection_data = self.collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        types = [metadata["doc_type"] for metadata in collection_data["metadatas"]]
        colors = [
            f"rgb({random.randint(50,220)}, {random.randint(50,220)}, {random.randint(50,220)})"
            for t in types
        ]
        tsne = TSNE(n_components=3 if self.is_3d else 2, random_state=42, perplexity=5)
        reduced_vectors = tsne.fit_transform(np.array(collection_data["embeddings"]))

        # Create the 2D scatter plot
        self.fig.add_trace(
            go.Scatter(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2] if self.is_3d else None,
                mode="markers",
                marker=dict(size=5, color=colors, opacity=0.8),
                text=[
                    f"Type: {t}<br>Text: {d[:100]}..."
                    for t, d in zip(types, collection_data["documents"])
                ],
                hoverinfo="text",
            )
        )

        self.fig.update_layout(
            title=(
                "3D Chroma Vector Store Visualization"
                if self.is_3d
                else "2D Chroma Vector Store Visualization"
            ),
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z" if self.is_3d else None,
            ),
        )

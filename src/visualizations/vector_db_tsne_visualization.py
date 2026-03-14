from chromadb import Collection
from visualizations.abstract_visualization import AbstractVisualization
import plotly.graph_objects as go
import random
from sklearn.manifold import TSNE
import numpy as np


class VectorDBTSNEVisualization(AbstractVisualization):

    def __init__(
        self,
        title: str,
        collection: Collection,
        is_3d=False,
        perplexity=5,
        show_question=True,
    ):
        super().__init__(title)
        self.collection = collection
        self.is_3d = is_3d
        self.perplexity = perplexity
        self.show_question = show_question
        self._highlight_ids: list[str] = []
        self._question: tuple[str, list[float]] = None

    @property
    def highlight_ids(self):
        return self._highlight_ids

    @highlight_ids.setter
    def highlight_ids(self, value: list[str]):
        # collection_data = self.collection.get()
        self._highlight_ids = value
        # self.fig.data[0].marker.symbol = [
        #     "square" if id in self._highlight_ids else "circle"
        #     for id in collection_data["ids"]
        # ]

    @property
    def question(self):
        return self._question

    @question.setter
    def question(self, value: tuple[str, list[float]]):
        self._question = value

    def draw(self):
        collection_data = self.collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        unique_types = list(
            {metadata["doc_type"] for metadata in collection_data["metadatas"]}
        )
        type_colors = {
            t: f"rgb({random.randint(50,220)}, {random.randint(50,220)}, {random.randint(50,220)})"
            for t in unique_types
        }
        types = [metadata["doc_type"] for metadata in collection_data["metadatas"]]

        colors = [
            type_colors[metadata["doc_type"]]
            for metadata in collection_data["metadatas"]
        ] + (["red"] if self._question and self.show_question else [])

        symbols = [
            "square" if id in self._highlight_ids else "circle"
            for id in collection_data["ids"]
        ] + (["diamond"] if self._question and self.show_question else [])

        texts = [
            f"Type: {t}<br>Text: {d[:100]}..."
            for t, d in zip(types, collection_data["documents"])
        ] + (
            [f"The question: {self._question[0]}"]
            if self._question and self.show_question
            else []
        )

        tsne = TSNE(
            n_components=3 if self.is_3d else 2,
            random_state=42,
            perplexity=self.perplexity,
        )
        embeddings = np.array(collection_data["embeddings"])
        if self._question:
            embeddings = np.vstack([embeddings, np.array([self._question[1]])])
        reduced_vectors = tsne.fit_transform(embeddings)

        if self.is_3d:
            self.fig.add_trace(
                go.Scatter3d(
                    x=reduced_vectors[:, 0],
                    y=reduced_vectors[:, 1],
                    z=reduced_vectors[:, 2],
                    mode="markers",
                    marker=dict(size=8, opacity=0.8, color=colors, symbol=symbols),
                    text=texts,
                    hoverinfo="text",
                )
            )
        else:
            self.fig.add_trace(
                go.Scatter(
                    x=reduced_vectors[:, 0],
                    y=reduced_vectors[:, 1],
                    mode="markers",
                    marker=dict(size=8, opacity=0.8, color=colors, symbol=symbols),
                    text=texts,
                    hoverinfo="text",
                )
            )

        self.fig.update_layout(
            title=(self.title + " (3D)" if self.is_3d else " (2D)"),
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z" if self.is_3d else None,
            ),
        )

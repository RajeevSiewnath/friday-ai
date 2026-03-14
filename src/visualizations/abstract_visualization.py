from abc import abstractmethod
import plotly.graph_objects as go


class AbstractVisualization:
    def __init__(self, title: str):
        self.title = title
        self.fig = go.Figure()

    @abstractmethod
    def draw(self):
        pass

    def get(self, redraw=True) -> go.Figure:
        if redraw:
            self.fig.data = tuple()
            self.draw()
        return self.fig

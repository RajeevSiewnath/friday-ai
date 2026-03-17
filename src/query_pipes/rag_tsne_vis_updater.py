from pipelines.pipeline import Pipe
from models.query_context import QueryContext, RagContext, RagContextCollection
from visualizations.vector_db_tsne_visualization import VectorDBTSNEVisualization


class RagTSNEVisUpdater(Pipe[QueryContext]):
    def __init__(self, vector_db_tsne_vis_updater: VectorDBTSNEVisualization):
        super().__init__()
        self.vector_db_tsne_vis_updater = vector_db_tsne_vis_updater

    def run(self, input):
        self.vector_db_tsne_vis_updater.highlight_ids = [
            context.id for context in input.context.contexts
        ]
        self.vector_db_tsne_vis_updater.question = (
            input.question,
            self.llm.embedding(input.question),
        )
        return input

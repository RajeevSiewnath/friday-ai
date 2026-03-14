from tqdm import tqdm
from pipelines.abstract_pipeline import AbstractPipe
from models.query_context import QueryContext, RagContext, RagContextCollection


class RagContextRetriever(AbstractPipe[QueryContext]):
    def __init__(self, n_results: int = 10, question_index: int = 0):
        super().__init__()
        self.n_results = n_results
        self.question_index = question_index

    def pipe(self, input):
        results = self.vector_db.query(
            input.question_history[self.question_index], n_results=self.n_results
        )
        input.context += RagContextCollection.from_contexts(
            [
                RagContext(
                    content=result.content, id=result.id, metadata=result.metadata
                )
                for result in tqdm(results.documents)
            ]
        )
        return input

from chromadb import Collection
from tqdm import tqdm
from pipelines.abstract_pipeline import AbstractPipe
from models.query_context import QueryContext, RagContext, RagContextCollection


class RagContextRetriever(AbstractPipe[QueryContext]):
    def __init__(self, retrieval_k: int = 10, question_index: int = 0):
        super().__init__()
        self.retrieval_k = retrieval_k
        self.question_index = question_index

    def pipe(self, arg):
        results = arg.vector_db.query(arg.input.question_history[self.question_index])
        arg.input.context += RagContextCollection.from_contexts(
            [
                RagContext(
                    content=result.content, id=result.id, metadata=result.metadata
                )
                for result in tqdm(results.documents)
            ]
        )
        return arg.input

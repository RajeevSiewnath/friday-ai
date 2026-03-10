from chromadb import Collection
from tqdm import tqdm
from pipelines.abstract_pipeline import AbstractPipe
from models.query_context import QueryContext, RagContext, RagContextCollection


class RagContextRetriever(AbstractPipe[QueryContext]):
    def __init__(
        self, retrieval_k: int = 10, question_index: int = 0
    ):
        super().__init__()
        self.retrieval_k = retrieval_k
        self.question_index = question_index

    def pipe(self, arg):
        query = arg.llm.embedding(arg.input.question_history[self.question_index])
        results = arg.vector_db.query(
            query_embeddings=query, n_results=self.retrieval_k
        )
        arg.input.context += RagContextCollection.from_contexts(
            [
                RagContext(content=result[0], id=result[1], metadata=result[2])
                for result in tqdm(
                    zip(
                        results["documents"][0],
                        results["ids"][0],
                        results["metadatas"][0],
                    )
                )
            ]
        )
        return arg.input

from chromadb import Collection
import chromadb
from tqdm import tqdm
from core.llm import embedding
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.query_pipeline import (
    QueryContext,
    RagContext,
    RagContextCollection,
)
from chromadb.config import Settings


class RagContextRetriever(AbstractPipe[QueryContext]):
    collection: Collection
    retrieval_k: int
    question_index: int

    def __init__(
        self, collection: Collection, retrieval_k: int = 10, question_index: int = 0
    ):
        super().__init__()
        self.collection = collection
        self.retrieval_k = retrieval_k
        self.question_index = question_index

    def pipe(self, input):
        query = embedding(input.question_history[self.question_index])
        results = self.collection.query(
            query_embeddings=query, n_results=self.retrieval_k
        )
        input.context += RagContextCollection.from_contexts(
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
        return input


if __name__ == "__main__":
    chroma = chromadb.Client(Settings(is_persistent=True))
    collection: Collection = chroma.get_collection(name="cv-rajeev-siewnath")
    query_optimizer: QueryContext = QueryContext(
        question_history=["where is javascript used?"],
        history=[{"role": "system", "content": "you are a kind agent"}],
    )
    print(RagContextRetriever(collection).pipe(query_optimizer))

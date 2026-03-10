from chromadb.config import Settings
from chromadb import Client, Collection
from core.llm import LLM
from models.document import Document, DocumentCollection


class VectorDB:
    def __init__(
        self,
        llm: LLM,
        name="vector-db",
    ):
        self.llm = llm
        self.chroma = Client(Settings(is_persistent=True))
        self.collection: Collection = self.chroma.get_or_create_collection(name=name)

    def populate(self, document_collection: DocumentCollection, force=False):
        if len(self.collection.get()["ids"]) == 0 or force:
            docs = document_collection.documents
            vectors = self.llm.embeddings([doc.content for doc in docs])
            self.collection.add(
                ids=[doc.id for doc in docs],
                embeddings=vectors,
                metadatas=[
                    {
                        **doc.metadata,
                        "doc_type": doc.type,
                        "file_path": doc.path,
                    }
                    for doc in docs
                ],
                documents=[doc.content for doc in docs],
            )
        return self

    def query(self, query: str, n_results=3):
        embedding = self.llm.embedding(query)
        results = self.collection.query(query_embeddings=embedding, n_results=n_results)
        return DocumentCollection.from_docs(
            [
                Document(
                    content=result[0],
                    id=result[1],
                    metadata=result[2],
                    type=result[2].get("doc_type", None),
                    path=result[2].get("file_path", None),
                )
                for result in zip(
                    results["documents"][0], results["ids"][0], results["metadatas"][0]
                )
            ]
        )

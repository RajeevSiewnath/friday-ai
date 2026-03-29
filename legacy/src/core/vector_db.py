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

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.collection is not None:
            self.delete()
        return False

    def populate(
        self, document_collection: DocumentCollection, force=False, clear=False
    ):
        if len(self.collection.get()["ids"]) == 0 or force:
            if clear:
                name = self.collection.name
                self.chroma.delete_collection(name)
                self.collection = self.chroma.create_collection(name)
            docs = document_collection.documents
            vectors = self.llm.embeddings([doc.content for doc in docs])
            self.collection.upsert(
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

    def delete(self):
        self.chroma.delete_collection(self.collection.name)
        self.collection = None

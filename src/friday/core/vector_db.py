from chromadb.config import Settings
from chromadb.api import ClientAPI
from chromadb import Client, Collection
from pydantic import BaseModel, Field


class VectorizableInstanceQuery(BaseModel):
    id: str = Field(description="The ID for the entry")
    metadata: dict = Field(description="The metadata for the entry")
    document: str = Field(description="The document for the entry")


class VectorizableInstanceInput(VectorizableInstanceQuery):
    embedding: list[float] = Field(description="The embeddings for the entry")


class VectorDB:
    def __init__(self, settings: Settings = Settings(is_persistent=True)):
        self.chroma: ClientAPI = Client(settings)
        self.__collections: list[Collection] = []

    # def __enter__(self):
    #     return self

    # def __exit__(self, _exc_type, _exc_val, _exc_tb):
    #     if self.collection is not None:
    #         self.delete()
    #     return False

    def __getitem__(self, key):
        collection = next(
            (collection for collection in self.__collections if collection.name == key),
            None,
        )
        if not collection:
            collection = self.chroma.get_or_create_collection(name=key)
            self.__collections.append(collection)

        return collection

    def delete(self, collection: Collection):
        self.__collections.remove(collection)
        self.chroma.delete_collection(collection)


class VectorDBCollection:
    def __init__(self, vector_db: VectorDB, collection: Collection):
        self.vector_db: VectorDB = vector_db
        self.collection: Collection = collection

    def populate(
        self, vectorizable: list[VectorizableInstanceInput], force=False, clear=False
    ):
        if len(self.collection.get()["ids"]) == 0 or force:
            if clear:
                name = self.collection.name
                self.vector_db.chroma.delete_collection(name)
                self.collection = self.vector_db.chroma.create_collection(name)
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            for i in vectorizable:
                ids.append(i.id)
                embeddings.append(i.embedding)
                metadatas.append(i.metadata)
                documents.append(i.document)
            self.collection.upsert(
                ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
            )
        return self

    def query(
        self, embedding: list[float], n_results=10
    ) -> list[VectorizableInstanceQuery]:
        results = self.collection.query(query_embeddings=embedding, n_results=n_results)
        return [
            VectorizableInstanceQuery(
                id=result[0], metadata=result[1], document=[result[2]]
            )
            for result in zip(
                results["ids"][0], results["metadatas"][0], results["documents"][0]
            )
        ]

    def delete(self):
        self.vector_db.delete(self.collection)

# if __name__ == "__main__":
#     chroma = chromadb.Client(Settings(is_persistent=True))
#     collection: Collection = chroma.get_collection(name="cv-rajeev-siewnath")
#     query_embedding = embedding("javascript")
#     query_optimizer: QueryContext = QueryContext(
#         context=collection.query(query_embeddings=query_embedding, n_results=3),
#         question_history=["where is javascript used?"],
#         history=[{"role": "system", "content": "you are a kind agent"}],
#     )
#     print(QueryRewriter().pipe(query_optimizer))

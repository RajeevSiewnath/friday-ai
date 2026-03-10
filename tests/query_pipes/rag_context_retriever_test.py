# if __name__ == "__main__":
#     chroma = chromadb.Client(Settings(is_persistent=True))
#     collection: Collection = chroma.get_collection(name="cv-rajeev-siewnath")
#     query_optimizer: QueryContext = QueryContext(
#         question_history=["where is javascript used?"],
#         history=[{"role": "system", "content": "you are a kind agent"}],
#     )
#     print(RagContextRetriever(collection).pipe(query_optimizer))

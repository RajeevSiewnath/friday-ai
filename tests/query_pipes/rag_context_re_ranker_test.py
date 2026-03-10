# if __name__ == "__main__":
#     chroma = chromadb.Client(Settings(is_persistent=True))
#     collection: Collection = chroma.get_collection(name="cv-rajeev-siewnath")
#     results = collection.get(limit=10)
#     rag_context = RagContextCollection.from_contexts(
#         [
#             RagContext(content=result[0], id=result[1], metadata=result[2])
#             for result in zip(
#                 results["documents"],
#                 results["ids"],
#                 results["metadatas"],
#             )
#         ]
#     )
#     query_optimizer: QueryContext = QueryContext(
#         question_history=["where is javascript used?"],
#         history=[{"role": "system", "content": "you are a kind agent"}],
#         context=rag_context,
#     )
#     print(RagContextReRanker().pipe(query_optimizer))

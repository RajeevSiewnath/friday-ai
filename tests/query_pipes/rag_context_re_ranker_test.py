from pydantic import BaseModel, Field
from core.llm import invoke
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.query_pipeline import QueryContext
from chromadb import Collection
import chromadb
from core.llm import embedding
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.query_pipeline import (
    QueryContext,
    RagContext,
    RagContextCollection,
)
from chromadb.config import Settings


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class RagContextReRanker(AbstractPipe[QueryContext]):
    def pipe(self, input):
        system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
        user_prompt = f"The user has asked the following question:\n\n{input.question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
        user_prompt += "Here are the chunks:\n\n"
        for index, chunk in enumerate(input.context.contexts):
            user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response: RankOrder = invoke(input=messages, response_format=RankOrder)
        input.context.contexts = [input.context.contexts[i - 1] for i in response.order]
        return input


if __name__ == "__main__":
    chroma = chromadb.Client(Settings(is_persistent=True))
    collection: Collection = chroma.get_collection(name="cv-rajeev-siewnath")
    results = collection.get(limit=10)
    rag_context = RagContextCollection.from_contexts(
        [
            RagContext(content=result[0], id=result[1], metadata=result[2])
            for result in zip(
                results["documents"],
                results["ids"],
                results["metadatas"],
            )
        ]
    )
    query_optimizer: QueryContext = QueryContext(
        question_history=["where is javascript used?"],
        history=[{"role": "system", "content": "you are a kind agent"}],
        context=rag_context,
    )
    print(RagContextReRanker().pipe(query_optimizer))

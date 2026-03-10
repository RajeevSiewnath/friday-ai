from pydantic import BaseModel, Field
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.query_pipeline import QueryContext
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.query_pipeline import (
    QueryContext,
)


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class RagContextReRanker(AbstractPipe[QueryContext]):
    def pipe(self, arg):
        system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
        user_prompt = f"The user has asked the following question:\n\n{arg.input.question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
        user_prompt += "Here are the chunks:\n\n"
        for index, chunk in enumerate(arg.input.context.contexts):
            user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = arg.llm.invoke(input=messages, response_format=RankOrder)
        arg.input.context.contexts = [
            arg.input.context.contexts[i - 1] for i in response.order
        ]
        return arg.input

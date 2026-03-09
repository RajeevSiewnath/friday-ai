from typing import Any
from chromadb import Collection
import chromadb
from chromadb.config import Settings
from core.llm import embedding, invoke
from core.prompt_context import PromptContext
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.query_pipeline import QueryContext


class QueryRewriter(AbstractPipe[QueryContext]):

    def rewrite_query(self, question: str, prompt_context: PromptContext):
        message = f"""You are in a conversation with a user, answering questions about {prompt_context.user_context_short}.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{prompt_context.history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention {prompt_context.user} unless it's a general question about {prompt_context.user}.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
        return invoke(input=[{"role": "system", "content": message}])

    def pipe(self, input):
        input.question = self.rewrite_query(input.question, input.history)
        return input


if __name__ == "__main__":
    chroma = chromadb.Client(Settings(is_persistent=True))
    collection: Collection = chroma.get_collection(name="cv-rajeev-siewnath")
    query_embedding = embedding("javascript")
    query_optimizer: QueryContext = QueryContext(
        context=collection.query(query_embeddings=query_embedding, n_results=3),
        question_history=["where is javascript used?"],
        history=[{"role": "system", "content": "you are a kind agent"}],
    )
    print(QueryRewriter().pipe(query_optimizer))
